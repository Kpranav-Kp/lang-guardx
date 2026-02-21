import torch
from pathlib import Path
from typing import Optional
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

# Path to saved model — relative to this file's location
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "langguardx_distilbert"

LABELS: list[str] = ["SAFE", "DANGEROUS", "INJECTION"]


class SQLIntentClassifier:
    """
    Fine-tuned DistilBERT classifier for SQL intent detection.
    Classifies input text as SAFE, DANGEROUS, or INJECTION.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,  
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
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}.\n"
                f"Run training/finetune_distilbert.py in Colab and place the "
                f"downloaded model folder at: {path}"
            )

        # Load tokenizer and model from local safetensors
        self.tokenizer: DistilBertTokenizerFast = (
            DistilBertTokenizerFast.from_pretrained(str(path))
        )

        self.model: DistilBertForSequenceClassification = (
            DistilBertForSequenceClassification.from_pretrained(
                str(path),
                local_files_only=True,
            )
        )
        self.model = self.model.to(self.device)  # type: ignore[assignment]
        self.model.eval()

    def predict(self, text: str) -> tuple[str, float]:
        """
        Run inference on a single input string.

        Returns:
            (label, confidence) e.g. ("DANGEROUS", 0.991)
        """
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
        pred_idx: int  = int(torch.argmax(probs).item())
        confidence: float = float(round(float(probs[pred_idx].item()), 4))
        label: str    = LABELS[pred_idx]

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

        probs: torch.Tensor    = torch.softmax(logits, dim=1)
        pred_idx: torch.Tensor = torch.argmax(probs, dim=1)

        results: list[tuple[str, float]] = []
        for i in range(len(texts)):
            idx: int    = int(pred_idx[i].item())
            conf: float = float(round(float(probs[i][idx].item()), 4))
            results.append((LABELS[idx], conf))

        return results