from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageClassification
from transformers.utils import logging as hf_logging
import tensorflow as tf

from inference.contracts import ImageSample

# Silencia los logs de UNEXPECTED keys
hf_logging.set_verbosity_error()


class ResNet50Pipeline:
    def __init__(
        self,
        model_folder_path: Path,
        labels_path: Path | None,
        device: torch.device
    ):
        self.model_folder_path = model_folder_path
        self.labels_path = labels_path
        self.device = device
        self.model = None
        self.labels = self._load_labels()
        self.image_size = 224
        
        # Normalización estándar para modelos preentrenados en ImageNet
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
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Se pidió GPU para un modelo PyTorch pero CUDA no está disponible en torch")

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


class RetinaNetPipeline:
    def __init__(
        self,
        model_folder_path: Path,
        labels_path: Path | None,
        device: torch.device
    ):
        self.model_folder_path = model_folder_path
        self.labels_path = labels_path
        self.device = device
        self.tf_device = "/CPU:0"
        self.model = None
        self.image_size = 800
        self.confidence_threshold = 0.5
        self.labels = self._load_labels()

    def _resolve_tf_device(self) -> str:
        if self.device.type == "cuda":
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                raise RuntimeError("Se pidió GPU pero TensorFlow no detecta ninguna GPU")
            return "/GPU:0"
        return "/CPU:0"

    def load(self) -> None:
        print(f"Cargando modelo desde: {self.model_folder_path}")
        if not self.model_folder_path.exists():
            raise FileNotFoundError(f"El modelo no se encontró en la ruta: {self.model_folder_path}")

        self.tf_device = self._resolve_tf_device()
        self.model = tf.saved_model.load(str(self.model_folder_path))
        print(f"Modelo cargado exitosamente en {self.tf_device}")

    def _image_to_tensor(self, sample: ImageSample) -> tf.Tensor:
        resized = sample.image.resize((self.image_size, self.image_size))
        image_array = np.array(resized, dtype=np.float32)
        return tf.convert_to_tensor(image_array)

    def preprocess(self, sample: ImageSample) -> dict[str, tf.Tensor]:
        print(f"Preprocesando muestra: {sample.path}")
        if self.model is None:
            raise RuntimeError("El modelo todavía no está cargado")
        
        tensor = self._image_to_tensor(sample)
        batch = tf.expand_dims(tensor, axis=0)
        return {"input_tensor": batch}

    def predict(self, model_inputs: dict[str, tf.Tensor]) -> dict:
        print("Ejecutando inferencia en el modelo")
        if self.model is None:
            raise RuntimeError("El modelo todavía no está cargado")

        concrete_func = self.model.signatures["serving_default"]
        with tf.device(self.tf_device):
            detections = concrete_func(input_1=model_inputs["input_tensor"])
        return detections

    def _load_labels(self) -> list[str]:
        if self.labels_path is None or not self.labels_path.exists():
            return []

        class_labels: list[str] = []
        with self.labels_path.open("r", encoding="utf-8") as labels_file:
            for raw_line in labels_file:
                line = raw_line.strip()
                if not line:
                    continue

                tokens = line.split(maxsplit=1)
                if len(tokens) == 2 and tokens[0].isdigit():
                    class_labels.append(tokens[1].strip())
                else:
                    class_labels.append(line)

        return class_labels

    def _resolve_label_for_class(self, class_id: int) -> str:
        if 0 <= class_id < len(self.labels):
            return self.labels[class_id]
        
        return f"class_{class_id}"

    def decode(self, logits: object, top_k: int = 5) -> list[dict[str, object]]:
        print("Decodificando resultados de inferencia")

        output_tensor = logits
        if isinstance(logits, dict):
            if not logits:
                return []
            output_tensor = next(iter(logits.values()))

        raw_output = np.asarray(output_tensor)
        if raw_output.ndim != 3 or raw_output.shape[0] == 0 or raw_output.shape[-1] <= 4:
            raise ValueError(f"Formato de salida no soportado en RetinaNet: shape={raw_output.shape}")

        # Formato esperado: (batch, anchors, 4 + num_classes)
        sample_output = raw_output[0]
        class_logits = sample_output[:, 4:]

        class_scores = 1.0 / (1.0 + np.exp(-class_logits))
        best_class_ids = np.argmax(class_scores, axis=1)
        best_scores = np.max(class_scores, axis=1)

        sorted_indices = np.argsort(-best_scores)

        predictions: list[dict[str, object]] = []
        seen_classes: set[int] = set()
        for anchor_idx in sorted_indices:
            class_id = int(best_class_ids[anchor_idx])
            if class_id in seen_classes:
                continue
            score = float(best_scores[anchor_idx])
            if score < self.confidence_threshold and predictions:
                break

            display_name = self._resolve_label_for_class(class_id)
            predictions.append(
                {
                    "class_index": class_id,
                    "label": display_name,
                    "score": score
                }
            )
            seen_classes.add(class_id)

            if len(predictions) >= top_k:
                break

        return predictions

    def infer(self, sample: ImageSample, top_k: int = 5) -> list[dict[str, object]]:
        print(f"Inferiendo muestra: {sample.path}")
        model_inputs = self.preprocess(sample)
        detections = self.predict(model_inputs)
        return self.decode(detections, top_k=top_k)


