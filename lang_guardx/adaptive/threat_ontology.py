from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AttackClass:
    id: str
    name: str
    boundary: str
    handled_by_layer: int
    detection_method: str
    update_target: str
    severity: str
    known_instances: list[str] = field(default_factory=list)
    fuzzer_generated: list[str] = field(default_factory=list)


class ThreatOntology:
    def __init__(self, yaml_path: str | None = None):
        path = Path(yaml_path) if yaml_path else Path(__file__).parent / "threat_taxonomy.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)

        self.version: str = data["version"]
        self.source: str = data["source"]
        self.attack_classes: dict[str, AttackClass] = {
            ac["id"]: AttackClass(
                **{
                    k: v
                    for k, v in ac.items()
                    if k != "fuzzer_generated"  # not in every entry
                }
            )
            for ac in data["attack_classes"]
        }

    def get_update_target(self, attack_id: str) -> str:
        if attack_id not in self.attack_classes:
            raise KeyError(f"Unknown attack id: {attack_id}")
        return self.attack_classes[attack_id].update_target

    def add_fuzzer_instance(self, attack_id: str, instance: str) -> None:
        if attack_id not in self.attack_classes:
            raise KeyError(f"Unknown attack id: {attack_id}")
        self.attack_classes[attack_id].fuzzer_generated.append(instance)

    def all_known_instances(self, attack_id: str) -> list[str]:
        """Returns static + runtime-learned instances for an attack class."""
        ac = self.attack_classes[attack_id]
        return ac.known_instances + ac.fuzzer_generated
