from pathlib import Path

import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import (
    DistilBertTokenizerFast,
)

DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "distilbert"

LABELS: list[str] = ["SAFE", "DANGEROUS", "INJECTION"]


class SQLIntentClassifier:
    """
    Fine-tuned DistilBERT classifier for SQL intent detection.
    Classifies input text as SAFE, DANGEROUS, or INJECTION.
    """

    def __init__(
        self,
        model_path: str | None = None,
        threshold: float = 0.75,
    ):
        """
        Args:
            model_path : Path to the saved model directory.
                         Defaults to models/langguardx_distilbert/
            threshold  : Minimum confidence to call something a threat.
                         Below this, SAFE is assumed even if model
                         leans toward a threat class. Default 0.75.
        """
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        onnx_path = path.parent / f"{path.name}_onnx"

        self._pred_cache = {}
        if not path.exists():
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id="KPranavKp/langguardx-distilbert", local_dir=str(path))

        # Load tokenizer and model from local safetensors
        self.tokenizer: DistilBertTokenizerFast = DistilBertTokenizerFast.from_pretrained(str(path))

        self.model = ORTModelForSequenceClassification.from_pretrained(
            str(onnx_path),
            provider="CPUExecutionProvider",
        )

    def predict(self, text: str) -> tuple[str, float]:
        """
        Run inference on a single input string.

        Returns:
            (label, confidence) e.g. ("DANGEROUS", 0.991)
        """
        key = text.strip().lower()
        if key in self._pred_cache:
            return self._pred_cache[key]
        if not text or not text.strip():
            return ("SAFE", 1.0)
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits: torch.Tensor = self.model(**enc).logits

        probs: torch.Tensor = torch.softmax(logits, dim=1)[0]
        pred_idx: int = int(torch.argmax(probs).item())
        confidence: float = float(round(float(probs[pred_idx].item()), 4))
        label: str = LABELS[pred_idx]

        self._pred_cache[key] = (label, confidence)
        return label, confidence

    def is_threat(self, text: str) -> bool:
        """
        Returns True if the input is classified as DANGEROUS or INJECTION
        with confidence >= threshold.

        This is the method the Detector orchestrator calls.
        """
        label, confidence = self.predict(text)
        return label != "SAFE" and confidence >= self.threshold

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """
        Run inference on a list of texts efficiently in one forward pass.
        Useful for the indirect scanner which may check multiple DB rows.
        """

        if not texts:
            return []
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits: torch.Tensor = self.model(**enc).logits

        probs: torch.Tensor = torch.softmax(logits, dim=1)
        pred_idx: torch.Tensor = torch.argmax(probs, dim=1)

        results: list[tuple[str, float]] = []
        for i in range(len(texts)):
            idx: int = int(pred_idx[i].item())
            conf: float = float(round(float(probs[i][idx].item()), 4))
            results.append((LABELS[idx], conf))

        return results

    def predict_proba(self, text: str) -> dict[str, float]:
        """Return probability distribution over LABELS."""
        if not text or not text.strip():
            return {"SAFE": 1.0, "DANGEROUS": 0.0, "INJECTION": 0.0}
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return {LABELS[i]: float(probs[i]) for i in range(3)}

    def decide(self, text: str, cost_fp: float = 1.0, cost_fn: float = 2.0, uncertain_ratio: float = 0.2, safe_override: bool = True) -> tuple[str, float]:
        """
        Returns (decision, margin) where decision is 'BLOCK', 'PASS', or 'UNCERTAIN'.
        If safe_override is True, text with no SQL keywords and no attack triggers is passed immediately.
        """
        if safe_override:
            # Quick heuristic: if no SQL keywords and no attack triggers, assume safe.
            sql_keywords = {"select", "from", "where", "join", "drop", "insert", "update", "delete", "create", "alter", "table", "database", "information_schema", "sqlite_master"}
            attack_triggers = {"ignore", "bypass", "override", "disregard", "forget", "reveal", "dump", "exfiltrate", "jailbreak", "unrestricted"}
            tokens = set(text.lower().split())
            if not (tokens & sql_keywords) and not (tokens & attack_triggers):
                return "PASS", 0.0

        # Original BRM logic
        probs = self.predict_proba(text)
        p_safe = probs["SAFE"]
        risk_block = cost_fp * p_safe
        risk_pass = cost_fn * (1 - p_safe)

        max_risk = max(risk_block, risk_pass)
        if max_risk == 0:
            return "PASS", 0.0

        relative_diff = abs(risk_block - risk_pass) / max_risk

        if relative_diff < uncertain_ratio:
            return "UNCERTAIN", relative_diff
        elif risk_block < risk_pass:
            return "BLOCK", (risk_pass - risk_block) / max_risk
        else:
            return "PASS", (risk_block - risk_pass) / max_risk
