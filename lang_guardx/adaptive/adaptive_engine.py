from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lang_guardx.adaptive.threat_ontology import ThreatOntology
from lang_guardx.detection.bloom import BloomDetector


class AdaptiveEngine:
    """Receives new attack patterns, classifies them by P2SQL taxonomy,
    and routes the update to the correct detection layer.

    Directly answers RQ4: the framework adapts to new patterns
    at runtime without restarting.
    """

    def __init__(
        self,
        bloom: BloomDetector,
        ontology: ThreatOntology,
        log_path: str = "adaptive_log.jsonl",
    ) -> None:
        self._bloom = bloom
        self._adaptive_bloom = BloomDetector()
        self._ontology = ontology
        self._log_path = Path(log_path)

    def add_pattern(self, attack_id: str, new_pattern: str) -> str:
        target = self._ontology.get_update_target(attack_id)

        if target == "bloom_corpus":
            self._adaptive_bloom.load_corpus([new_pattern], min_window=4)
            self._ontology.add_fuzzer_instance(attack_id, new_pattern)
            self._log(attack_id, new_pattern, target)

        elif target == "none":
            pass

        return target

    def get_adaptive_bloom(self) -> BloomDetector:
        """Return the adaptive Bloom filter for use in the detector."""
        return self._adaptive_bloom

    def get_adaptation_count(self) -> int:
        """Total patterns learned at runtime — thesis RQ4 metric."""
        if not self._log_path.exists():
            return 0
        with open(self._log_path) as f:
            return sum(1 for _ in f)

    def get_adaptation_log(self) -> list[dict]:
        """Return full log as list of dicts for reporting."""
        if not self._log_path.exists():
            return []
        with open(self._log_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def export_state(self, path: str | Path) -> None:
        """Export runtime state (adaptive bloom filter state + adaptation log) to JSON."""
        state = {
            "version": 1,
            "exported_at": datetime.now(UTC).isoformat(),
            "adaptation_log": self.get_adaptation_log(),
            "adaptive_bloom_signatures": self._adaptive_bloom.signature_count,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def import_state(self, path: str | Path) -> None:
        """Import previously exported runtime state (replays adaptation log)."""
        with open(path) as f:
            state = json.load(f)
        for entry in state.get("adaptation_log", []):
            self.add_pattern(entry["attack_id"], entry["pattern"])

    def _log(self, attack_id: str, pattern: str, target: str) -> None:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "attack_id": attack_id,
            "pattern": pattern,
            "routed_to": target,
        }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
