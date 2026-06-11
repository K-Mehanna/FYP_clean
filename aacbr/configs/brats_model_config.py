import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BraTSOutcomeConfig:
    strategy: str
    include_supports: bool
    supported_attack_chain: bool
    default_outcome: int
    default_class: int
    num_kmeans_centroids: int | None = None
    threshold: float | None = None

    @classmethod
    def from_json(cls, json_path: str | Path) -> "BraTSOutcomeConfig":
        raw = json.loads(Path(json_path).read_text())
        return cls(**raw)
