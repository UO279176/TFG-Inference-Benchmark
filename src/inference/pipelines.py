from pathlib import Path
from typing import Any, cast
from datetime import datetime
import os

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from transformers import AutoModelForImageClassification, AutoModelForCausalLM, LlamaTokenizer
from transformers.utils import logging as hf_logging
import tensorflow as tf
from nemo.collections.asr.models import ASRModel
from rknnlite.api import RKNNLite

from inference.contracts import AudioSample, ImageSample, TextSample
from data import ExecutionTarget, Accelerator

# Silencia los logs de UNEXPECTED keys
hf_logging.set_verbosity_error()

# MARK: Resnet50
class ResNet50Pipeline:
    def __init__(
        self,
        model_folder_path: Path,
        labels_path: Path | None,
        target: ExecutionTarget
    ):
        self.model_folder_path = model_folder_path
        self.labels_path = labels_path
        self.target = target
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
        if self.target.accelerator == Accelerator.GPU and not torch.cuda.is_available():
            raise RuntimeError("Se pidió GPU para un modelo PyTorch pero CUDA no está disponible en torch")

        if self.target.accelerator == Accelerator.CPU or self.target.accelerator == Accelerator.GPU:
            self.model = AutoModelForImageClassification.from_pretrained(str(self.model_folder_path))
            self.model.to(self.target.device)
            self.model.eval()
            
        elif self.target.accelerator == Accelerator.NPU:
            self.model = RKNNLite()
            ret = self.model.load_rknn(self.model_folder_path / "resnet50.rknn")
            if ret != 0:
                raise RuntimeError(f"Error al cargar el modelo RKNN: código de error {ret}")
            
            ret = self.model.init_runtime()
            if ret != 0:
                raise RuntimeError(f"Error al inicializar el runtime RKNN: código de error {ret}")
            
        else:
            raise RuntimeError(f"Acelerador no soportado para ResNet50: {self.target.accelerator}")
        
        print("Modelo cargado exitosamente")

    def _image_to_tensor(self, sample: ImageSample) -> torch.Tensor:
        resized = sample.image.resize((self.image_size, self.image_size))
        image_bytes = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
        tensor = image_bytes.view(self.image_size, self.image_size, 3).permute(2, 0, 1).float() / 255.0
        normalized = (tensor - self.mean) / self.std
        return normalized
    
    def _image_to_numpy(self, sample: ImageSample) -> np.ndarray:
        resized = sample.image.resize((self.image_size, self.image_size))
        image_array = np.asarray(resized, dtype=np.float32)
        normalized = (image_array / 255.0 - self.mean.numpy().transpose(1, 2, 0)) / self.std.numpy().transpose(1, 2, 0)
        return np.ascontiguousarray(normalized[None, ...])

    def preprocess(self, sample: ImageSample) -> dict[str, Any]:
        print(f"Preprocesando muestra: {sample.path}")
        if self.model is None:
            raise RuntimeError("El modelo todavía no está cargado")

        pixel_values = None
        if self.target.accelerator == Accelerator.CPU or self.target.accelerator == Accelerator.GPU:
            pixel_values = self._image_to_tensor(sample).unsqueeze(0).to(self.target.device)
        elif self.target.accelerator == Accelerator.NPU:
            pixel_values = self._image_to_numpy(sample)
        else:
            raise RuntimeError(f"Acelerador no soportado para ResNet50: {self.target.accelerator}")

        return {"pixel_values": pixel_values}

    def predict(self, model_inputs: dict[str, Any]) -> torch.Tensor:
        print("Ejecutando inferencia en el modelo")
        if self.model is None:
            raise RuntimeError("El modelo todavía no está cargado")

        if self.target.accelerator == Accelerator.CPU or self.target.accelerator == Accelerator.GPU:
            with torch.inference_mode():
                return self.model(**model_inputs).logits
            
        elif self.target.accelerator == Accelerator.NPU:
            rknn_outputs = self.model.inference(inputs=[model_inputs["pixel_values"]])
            if not rknn_outputs:
                raise RuntimeError("RKNN no devolvió salidas")

            raw_output = rknn_outputs[0]
            if isinstance(raw_output, torch.Tensor):
                return raw_output

            if isinstance(raw_output, np.ndarray):
                return torch.from_numpy(raw_output)

            return torch.as_tensor(raw_output)
        
        else:
            raise RuntimeError(f"Acelerador no soportado para ResNet50: {self.target.accelerator}")

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

    def unload(self) -> None:
        print("Descargando modelo de memoria")
        if self.model is None:
            print("El modelo ya estaba descargado")
            return
        
        if self.target.accelerator == Accelerator.NPU:
            self.model.release()
            

# MARK: RetinaNet
class RetinaNetPipeline:
    def __init__(
        self,
        model_folder_path: Path,
        labels_path: Path | None,
        target: ExecutionTarget
    ):
        self.model_folder_path = model_folder_path
        self.labels_path = labels_path
        self.target = target
        self.tf_device = "/CPU:0"
        self.model = None
        self.image_size = 800
        self.confidence_threshold = 0.5
        self.labels = self._load_labels()
        
        # Normalización estándar para RetinaNet preentrenado en COCO
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def _resolve_tf_device(self) -> str:
        if self.target.accelerator == Accelerator.GPU:
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                raise RuntimeError("Se pidió GPU pero TensorFlow no detecta ninguna GPU")
            return "/GPU:0"
        return "/CPU:0"

    def load(self) -> None:
        print(f"Cargando modelo desde: {self.model_folder_path}")
        if not self.model_folder_path.exists():
            raise FileNotFoundError(f"El modelo no se encontró en la ruta: {self.model_folder_path}")
        
        if self.target.accelerator == Accelerator.CPU or self.target.accelerator == Accelerator.GPU:
            self.tf_device = self._resolve_tf_device()
            self.model = tf.saved_model.load(str(self.model_folder_path))
            print(f"Modelo cargado exitosamente en {self.tf_device}")
            
        elif self.target.accelerator == Accelerator.NPU:
            self.model = RKNNLite()
            ret = self.model.load_rknn(self.model_folder_path / "retinanet.rknn")
            if ret != 0:
                raise RuntimeError(f"Error al cargar el modelo RKNN: código de error {ret}")
            
            ret = self.model.init_runtime()
            if ret != 0:
                raise RuntimeError(f"Error al inicializar el runtime RKNN: código de error {ret}")
            
        else:
            raise RuntimeError(f"Acelerador no soportado para RetinaNet: {self.target.accelerator}")

    def _image_to_tensor(self, sample: ImageSample) -> tf.Tensor:
        resized = sample.image.resize((self.image_size, self.image_size))
        image_array = np.array(resized, dtype=np.float32)
        return tf.convert_to_tensor(image_array)

    def _image_to_numpy(self, sample: ImageSample) -> np.ndarray:
        resized = sample.image.resize((self.image_size, self.image_size))
        image_array = np.asarray(resized, dtype=np.float32)
        image_array = image_array * self.std + self.mean
        image_array = np.transpose(image_array, (2, 0, 1))
        return np.ascontiguousarray(image_array[None, ...])

    def preprocess(self, sample: ImageSample) -> dict[str, Any]:
        print(f"Preprocesando muestra: {sample.path}")
        if self.model is None:
            raise RuntimeError("El modelo todavía no está cargado")
        
        batch = None
        if self.target.accelerator == Accelerator.CPU or self.target.accelerator == Accelerator.GPU:
            tensor = self._image_to_tensor(sample)
            batch = tf.expand_dims(tensor, axis=0)
        elif self.target.accelerator == Accelerator.NPU:
            batch = self._image_to_numpy(sample)
        else:
            raise RuntimeError(f"Acelerador no soportado para RetinaNet: {self.target.accelerator}")
            
        return {"input_tensor": batch}

    def predict(self, model_inputs: dict[str, Any]) -> dict:
        print("Ejecutando inferencia en el modelo")
        if self.model is None:
            raise RuntimeError("El modelo todavía no está cargado")

        if self.target.accelerator == Accelerator.CPU or self.target.accelerator == Accelerator.GPU:
            concrete_func = self.model.signatures["serving_default"]
            with tf.device(self.tf_device):
                detections = concrete_func(input_1=model_inputs["input_tensor"])
            return detections
        
        elif self.target.accelerator == Accelerator.NPU:
            rknn_outputs = self.model.inference(inputs=[model_inputs["input_tensor"]])
            if not rknn_outputs:
                raise RuntimeError("RKNN no devolvió salidas")
            return rknn_outputs
        
        else:
            raise RuntimeError(f"Acelerador no soportado para RetinaNet: {self.target.accelerator}")

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

    def _normalize_retinanet_output(self, raw_output: np.ndarray) -> np.ndarray:
        '''
        Normaliza la salida de RetinaNet para que tenga la forma (N, 84), donde N es el número de predicciones.
        '''
        output = np.asarray(raw_output)
        output = np.squeeze(output)

        # Si la salida es un tensor 3D, verificamos si la última dimensión es 84 (número de clases + 4 coordenadas de caja)
        if output.ndim == 3:
            if output.shape[-1] == 84:
                return output[0] if output.shape[0] == 1 else output

            if output.shape[1] == 84:
                transposed = np.transpose(output, (0, 2, 1))
                return transposed[0] if transposed.shape[0] == 1 else transposed

        # Si la salida es un tensor 2D, verificamos si alguna de las dimensiones es 84
        if output.ndim == 2:
            if output.shape[-1] == 84:
                return output

            if output.shape[0] == 84:
                return output.T

        # Si la salida es un tensor 1D, verificamos si su tamaño es múltiplo de 84
        if output.ndim == 1 and output.size % 84 == 0:
            return output.reshape((-1, 84))

        raise ValueError(f"Formato de salida no soportado en RetinaNet: shape={output.shape}")

    def decode(self, logits: object, top_k: int = 5) -> list[dict[str, object]]:
        print("Decodificando resultados de inferencia")

        output_tensor = logits
        if isinstance(logits, dict):
            if not logits:
                return []
            output_tensor = next(iter(logits.values()))

        raw_output = np.asarray(output_tensor)
        
        sample_output = self._normalize_retinanet_output(raw_output)
        if sample_output.shape[-1] <= 4:
            raise ValueError(f"Formato de salida no soportado en RetinaNet: shape={sample_output.shape}")

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

    def unload(self) -> None:
        print("Descargando modelo de memoria")
        if self.model is None:
            print("El modelo ya estaba descargado")
            return
        
        if self.target.accelerator == Accelerator.NPU:
            self.model.release()
            

# MARK: TinyLlama
class TinyLlamaPipeline:
    def __init__(
        self,
        model_folder_path: Path,
        labels_path: Path | None,
        target: ExecutionTarget
    ):
        self.model_folder_path = model_folder_path
        self.target = target
        self.model: Any = None
        self.tokenizer: Any = None
        self.max_input_chars = 1200
        self.max_output_tokens = 50

    def load(self) -> None:
        print(f"Cargando modelo desde: {self.model_folder_path}")
        if not self.model_folder_path.exists():
            raise FileNotFoundError(f"El modelo no se encontró en la ruta: {self.model_folder_path}")

        try:
            self.tokenizer = LlamaTokenizer.from_pretrained(
                str(self.model_folder_path),
                local_files_only=True,
            )
        except Exception as tokenizer_error:
            raise RuntimeError(f"Error al cargar el tokenizer de TinyLlama: {tokenizer_error}")

        # Configurar pad token si no existe
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_folder_path),
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        self.model.to(self.target.device)
        self.model.eval()
        print("Modelo cargado exitosamente")

    def preprocess(self, sample: TextSample) -> dict[str, Any]:
        print(f"Preprocesando muestra: {sample.path}")
        tokenizer = self.tokenizer # Guardamos en variable local para evitar problemas de acceso
        if tokenizer is None:
            raise RuntimeError("El modelo todavía no está cargado")

        prompt_text = sample.prompt[:self.max_input_chars]
        prompt = f"Summarize the following news article:\n\n{prompt_text}"
        tokenized_inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(self.target.device)
        input_length = tokenized_inputs["input_ids"].shape[1]
        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "input_length": torch.tensor([input_length], device=self.target.device)
        }

    def predict(self, model_inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        print("Ejecutando inferencia en el modelo")
        model = self.model # Guardamos en variable local para evitar problemas de acceso
        tokenizer = self.tokenizer
        if model is None or tokenizer is None:
            raise RuntimeError("El modelo todavía no está cargado")

        with torch.inference_mode():
            output_ids = model.generate(
                model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                min_new_tokens=8,
                max_new_tokens=self.max_output_tokens,
                temperature=0, # Determinista
                do_sample=False, # Greedy decoding
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=False,
            )

        if not isinstance(output_ids, torch.Tensor):
            raise RuntimeError("TinyLlama devolvió un formato de generación no soportado")

        return {
            "output_ids": output_ids,
            "input_length": model_inputs["input_length"]
        }

    def decode(self, logits: dict[str, Any], top_k: int = 5) -> list[dict[str, object]]:
        print("Decodificando resultados de inferencia")
        tokenizer = self.tokenizer # Guardamos en variable local para evitar problemas de acceso
        if tokenizer is None:
            raise RuntimeError("El modelo todavía no está cargado")

        if "output_ids" not in logits or "input_length" not in logits:
            raise ValueError("Formato de salida no soportado en TinyLlama")

        output_ids = logits["output_ids"]
        input_length = int(logits["input_length"][0].item())

        continuation_ids = output_ids[0][input_length:]
        continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()

        if continuation_text:
            display_text = continuation_text
        else:
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            display_text = full_text if full_text else "[sin texto generado]"

        predictions: list[dict[str, object]] = [
            {
                "text": display_text
            }
        ]

        return predictions

    def infer(self, sample: TextSample, top_k: int = 5) -> list[dict[str, object]]:
        print(f"Inferiendo muestra: {sample.path}")
        model_inputs = self.preprocess(sample)
        output_ids = self.predict(model_inputs)
        return self.decode(output_ids, top_k=top_k)

    def unload(self) -> None:
        print("Descargando modelo de memoria")
        if self.model is None:
            print("El modelo ya estaba descargado")
            return
        
        if self.target.accelerator == Accelerator.NPU:
            self.model.release()
            

# MARK: Stable Diffusion 1.5
class StableDiffusion15Pipeline:
    def __init__(
        self,
        model_folder_path: Path,
        labels_path: Path | None,
        target: ExecutionTarget
    ):
        self.model_folder_path = model_folder_path
        self.target = target
        self.pipeline = None
        self.num_inference_steps = 8 # A mayor cantidad de pasos mejora calidad pero aumenta tiempo de inferencia
        self.guidance_scale = 7.5 # A mayor guidance scale, más se adhiere la generación al prompt pero puede perder creatividad
        self.height = 256
        self.width = 256
        self.seed = 11
        self.output_folder_path = self.model_folder_path / "generated_images"
        self.generation_count = 0 # Contador para evitar colisiones de nombres en las imágenes generadas

    def load(self) -> None:
        print(f"Cargando modelo desde: {self.model_folder_path}")
        if not self.model_folder_path.exists():
            raise FileNotFoundError(f"El modelo no se encontró en la ruta: {self.model_folder_path}")

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            str(self.model_folder_path),
            torch_dtype=torch.float32,
            local_files_only=True,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )

        self.pipeline.to(self.target.device)
        self.pipeline.enable_attention_slicing()
        self.pipeline.set_progress_bar_config(disable=True)
        self.output_folder_path.mkdir(parents=True, exist_ok=True)
        print("Modelo cargado exitosamente")

    def preprocess(self, sample: TextSample) -> dict[str, object]:
        print(f"Preprocesando muestra: {sample.path}")
        if self.pipeline is None:
            raise RuntimeError("El modelo todavía no está cargado")

        assert self.target.device is not None
        generator = torch.Generator(device=self.target.device.type).manual_seed(self.seed)
        
        return {
            "sample_path": sample.path,
            "prompt": sample.prompt,
            "generator": generator,
        }

    def predict(self, model_inputs: dict[str, object]) -> dict[str, object]:
        print("Ejecutando inferencia en el modelo")
        if self.pipeline is None:
            raise RuntimeError("El modelo todavía no está cargado")

        prompt: Any = model_inputs["prompt"]
        generator: Any = model_inputs["generator"]

        result: Any = self.pipeline(
            prompt=prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            height=self.height,
            width=self.width,
            generator=generator,
        )

        image: Any = result.images[0]
        return {
            "sample_path": model_inputs["sample_path"],
            "prompt": prompt,
            "image": image,
        }

    def decode(self, logits: dict[str, object], top_k: int = 5) -> list[dict[str, object]]:
        print("Decodificando resultados de inferencia")
        if "image" not in logits or "prompt" not in logits or "sample_path" not in logits:
            raise ValueError("Formato de salida no soportado en Stable Diffusion")

        image: Any = logits["image"]
        if not hasattr(image, "size") or not hasattr(image, "save"):
            raise ValueError("La salida de Stable Diffusion no contiene una imagen válida")

        sample_path = logits["sample_path"]
        sample_name = str(sample_path)
        safe_sample_name = "".join(
            character if character.isalnum() or character in ("-", "_") else "_"
            for character in sample_name
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.generation_count += 1
        output_filename = (
            f"{safe_sample_name}_{timestamp}_{self.generation_count:03d}_"
            f"{self.width}x{self.height}.png"
        )
        output_path = self.output_folder_path / output_filename
        image.save(output_path)

        width, height = image.size
        return [
            {
                "sample_path": logits["sample_path"],
                "prompt": logits["prompt"],
                "width": width,
                "height": height,
                "saved_path": str(output_path),
            }
        ]

    def infer(self, sample: TextSample, top_k: int = 5) -> list[dict[str, object]]:
        print(f"Inferiendo muestra: {sample.path}")
        model_inputs = self.preprocess(sample)
        output = self.predict(model_inputs)
        return self.decode(output, top_k=top_k)

    def unload(self) -> None:
        print("Descargando modelo de memoria")
        if self.pipeline is None:
            print("El modelo ya estaba descargado")
            return
        
        if self.target.accelerator == Accelerator.NPU:
            self.pipeline.release()
            

# MARK: RNNT
class RNNTPipeline:
    def __init__(
        self,
        model_folder_path: Path,
        labels_path: Path | None,
        target: ExecutionTarget
    ):
        self.model_folder_path = model_folder_path
        self.target = target
        self.model = None

    def _resolve_model_path(self) -> Path:
        nemo_files = sorted(self.model_folder_path.glob("*.nemo"))
        if not nemo_files:
            raise FileNotFoundError(
                f"No se encontró ningún archivo .nemo en: {self.model_folder_path}"
            )
        return nemo_files[0]

    def load(self) -> None:
        print(f"Cargando modelo desde: {self.model_folder_path}")
        if not self.model_folder_path.exists():
            raise FileNotFoundError(f"El modelo no se encontró en la ruta: {self.model_folder_path}")
        if self.target.accelerator == Accelerator.CPU:
            # Para evitar que NeMo intente usar GPU cuando se ha especificado CPU, deshabilitamos la visibilidad de las GPUs a nivel de entorno
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"

        model_path = self._resolve_model_path()
        loaded_model: Any = ASRModel.restore_from(
            restore_path=str(model_path),
            map_location=self.target.device,
        )
        self.model = loaded_model
        model: Any = self.model
        model.eval()
        print("Modelo cargado exitosamente")

    def preprocess(self, sample: AudioSample) -> dict[str, object]:
        print(f"Preprocesando muestra: {sample.path}")
        if self.model is None:
            raise RuntimeError("El modelo todavía no está cargado")

        if not sample.audio_path.exists():
            raise FileNotFoundError(f"No se encontró el audio en la ruta: {sample.audio_path}")

        return {
            "audio_path": str(sample.audio_path),
            "reference": sample.reference,
        }

    def _normalize_transcription(self, raw_output: object) -> str:
        if isinstance(raw_output, list) and raw_output:
            first_output = raw_output[0]
            if isinstance(first_output, str):
                return first_output.strip()
            if hasattr(first_output, "text"):
                return str(first_output.text).strip()

        if isinstance(raw_output, str):
            return raw_output.strip()

        return str(raw_output).strip()

    def predict(self, model_inputs: dict[str, object]) -> dict[str, object]:
        print("Ejecutando inferencia en el modelo")
        model: Any = self.model
        if model is None:
            raise RuntimeError("El modelo todavía no está cargado")

        with torch.inference_mode():
            raw_transcription = model.transcribe(
                audio=[model_inputs["audio_path"]],
                use_lhotse=False,
                batch_size=1,
            )

        transcription = self._normalize_transcription(raw_transcription)
        return {
            "text": transcription,
            "reference": model_inputs.get("reference"),
        }

    def decode(self, logits: dict[str, object], top_k: int = 5) -> list[dict[str, object]]:
        print("Decodificando resultados de inferencia")
        if "text" not in logits:
            raise ValueError("Formato de salida no soportado en RNNT")

        text = str(logits["text"]).strip()
        if not text:
            text = "[sin transcripción]"

        return [
            {
                "text": text,
                "reference": logits.get("reference"),
            }
        ]

    def infer(self, sample: AudioSample, top_k: int = 5) -> list[dict[str, object]]:
        print(f"Inferiendo muestra: {sample.path}")
        model_inputs = self.preprocess(sample)
        output = self.predict(model_inputs)
        return self.decode(output, top_k=top_k)

    def unload(self) -> None:
        print("Descargando modelo de memoria")
        if self.model is None:
            print("El modelo ya estaba descargado")
            return
        
        if self.target.accelerator == Accelerator.NPU:
            self.model.release()