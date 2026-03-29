from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from PIL import Image


@dataclass(frozen=True)
class ImageSample:
    path: Path
    image: Image.Image


class DatasetAdapter(Protocol):
    def iter_samples(self, limit: int | None = None) -> list[ImageSample]:
        """Iterar sobre las muestras del dataset y devolver una lista de ImageSample."""
        ...


class ModelPipeline(Protocol):
    def load(self) -> None:
        """Cargar el modelo y cualquier recurso necesario (ej. etiquetas)."""
        ...

    def preprocess(self, sample: ImageSample) -> dict[str, torch.Tensor]:
        """Convertir la imagen de entrada en los tensores que el modelo espera como input."""
        ...

    def predict(self, model_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Ejecutar la inferencia en el modelo y devolver los logits (salidas crudas)."""
        ...

    def decode(self, logits: torch.Tensor, top_k: int = 5) -> list[tuple[int, float, str]]:
        """Convertir los logits en una lista de tuplas (class_index, score, label) para las top_k predicciones."""
        ...

    def infer(self, sample: ImageSample, top_k: int = 5) -> list[tuple[int, float, str]]:
        """Ejecutar el pipeline completo de inferencia: preprocess -> predict -> decode."""
        ...
