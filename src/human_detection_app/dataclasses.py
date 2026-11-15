from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class HumanImageSample:
    path: Path
    human_count: int # Solo para benchmarking, no existiría en un sistema de producción
    bboxes: list # Solo para benchmarking, no existiría en un sistema de producción
    
@dataclass(frozen=True)
class InferenceResult:
    locations: list
    categories: list
    scores: list
    number: int