from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class HumanImageSample:
    path: Path
    human_present: bool
    
@dataclass(frozen=True)
class InferenceResult:
    locations: list
    categories: list
    scores: list
    number: int