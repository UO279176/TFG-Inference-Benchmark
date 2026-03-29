from pathlib import Path

import torch
from transformers import AutoModelForImageClassification
from transformers.utils import logging as hf_logging

from inference.contracts import ImageSample

# Silencia los logs de UNEXPECTED keys
hf_logging.set_verbosity_error()


class ResNet50Pipeline:
    def __init__(self, model_folder_path: Path, labels_path: Path | None, device: torch.device):
        self.model_folder_path = model_folder_path
        self.labels_path = labels_path
        self.device = device
        self.model = None
        self.labels = self._load_labels()
        self.image_size = 224
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def _load_labels(self) -> list[str]:
        if self.labels_path is None or not self.labels_path.exists():
            return []

        with self.labels_path.open("r", encoding="utf-8") as labels_file:
            return [line.strip() for line in labels_file if line.strip()]

    def load(self) -> None:
        print(f"Cargando modelo desde: {self.model_folder_path}")
        if not self.model_folder_path.exists():
            raise FileNotFoundError(f"El modelo no se encontró en la ruta: {self.model_folder_path}")

        self.model = AutoModelForImageClassification.from_pretrained(str(self.model_folder_path))
        self.model.to(self.device)
        self.model.eval()

    def _image_to_tensor(self, sample: ImageSample) -> torch.Tensor:
        resized = sample.image.resize((self.image_size, self.image_size))
        image_bytes = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
        tensor = image_bytes.view(self.image_size, self.image_size, 3).permute(2, 0, 1).float() / 255.0
        normalized = (tensor - self.mean) / self.std
        return normalized

    def preprocess(self, sample: ImageSample) -> dict[str, torch.Tensor]:
        print(f"Preprocesando muestra: {sample.path}")
        if self.model is None:
            raise RuntimeError("El modelo todavía no está cargado")

        pixel_values = self._image_to_tensor(sample).unsqueeze(0).to(self.device)
        return {"pixel_values": pixel_values}

    def predict(self, model_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        print("Ejecutando inferencia en el modelo")
        if self.model is None:
            raise RuntimeError("El modelo todavía no está cargado")

        with torch.inference_mode():
            return self.model(**model_inputs).logits

    def decode(self, logits: torch.Tensor, top_k: int = 5) -> list[tuple[int, float, str]]:
        print("Decodificando resultados de inferencia")
        probabilities = torch.softmax(logits[0], dim=-1)
        scores, indices = torch.topk(probabilities, k=top_k)

        predictions: list[tuple[int, float, str]] = []
        for class_index, score in zip(indices.tolist(), scores.tolist()):
            label = self.labels[class_index] if class_index < len(self.labels) else f"class_{class_index}"
            predictions.append((class_index, score, label))

        return predictions

    def infer(self, sample: ImageSample, top_k: int = 5) -> list[tuple[int, float, str]]:
        print(f"Inferiendo muestra: {sample.path}")
        model_inputs = self.preprocess(sample)
        logits = self.predict(model_inputs)
        return self.decode(logits, top_k=top_k)
