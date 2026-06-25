from __future__ import annotations

import faulthandler
faulthandler.enable()

from pathlib import Path
from typing import Any, cast
from datetime import datetime
import ctypes
import os
import tempfile

import numpy as np
import json
import torch
import torchaudio
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from transformers import AutoModelForImageClassification, AutoModelForCausalLM, LlamaTokenizerFast, CLIPTokenizer
from transformers.utils import logging as hf_logging
import tensorflow as tf
from diffusers.schedulers import LCMScheduler

from inference.contracts import AudioSample, ImageSample, TextSample
from data import ExecutionTarget, Accelerator

try:
    from nemo.collections.asr.models import ASRModel
except ImportError:
    ASRModel = None
    print("Advertencia: nemo_toolkit no está instalado o no es compatible con la versión de Python. Los modelos NeMo para CPU/GPU no funcionarán.")

try:
    from rknnlite.api import RKNNLite
    from inference.rkllm_tl_lib.rkllm import RKLLM
    from inference.rkllm_tl_lib.variables import global_text
    from inference.rknn_sd_lib.rknn_sd import RKNN2LatentConsistencyPipeline, RKNN2Model
except ImportError:
    RKNNLite = None
    RKLLM = None
    RKNN2LatentConsistencyPipeline = None
    RKNN2Model = None
    print("Advertencia: rknnlite y/o la librería no están instaladas. Los modelos RKNN y/o RKLLM para NPU no funcionarán.")
    
try:
    from tflite_runtime.interpreter import load_delegate, Interpreter
except ImportError:
    tflite = None
    print("Advertencia: tflite_runtime no está instalado. Los modelos TFLite para TPU no funcionarán.")

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
            
        elif self.target.accelerator == Accelerator.TPU:
            delegate = load_delegate("libedgetpu.so.1")
            
            self.model = Interpreter(
                model_path=str(self.model_folder_path / "tfhub_tf2_resnet_50_imagenet_ptq_edgetpu.tflite"),
                experimental_delegates=[delegate]
            )
            self.model.allocate_tensors()
   
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

    def _image_to_tpu(self, sample: ImageSample) -> np.ndarray:
        resized = sample.image.resize((self.image_size, self.image_size))
        image_array = np.array(resized, dtype=np.uint8)
        input_data = np.expand_dims(image_array, axis=0)
        return input_data

    def preprocess(self, sample: ImageSample) -> dict[str, Any]:
        print(f"Preprocesando muestra: {sample.path}")
        if self.model is None:
            raise RuntimeError("El modelo todavía no está cargado")

        pixel_values = None
        if self.target.accelerator == Accelerator.CPU or self.target.accelerator == Accelerator.GPU:
            pixel_values = self._image_to_tensor(sample).unsqueeze(0).to(self.target.device)
        elif self.target.accelerator == Accelerator.NPU:
            pixel_values = self._image_to_numpy(sample)
        elif self.target.accelerator == Accelerator.TPU:
            pixel_values = self._image_to_tpu(sample)
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

        elif self.target.accelerator == Accelerator.TPU:
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()

            self.model.set_tensor(input_details[0]['index'], model_inputs["pixel_values"])
            self.model.invoke()
            output_data = self.model.get_tensor(output_details[0]['index'])
            
            scale, zero_point = output_details[0]['quantization']
            
            # Convertir a float32 y aplicar la fórmula de decuantización
            # Fórmula: real_value = (quantized_value - zero_point) * scale
            if scale > 0:
                output_data = (output_data.astype(np.float32) - zero_point) * scale
            else:
                output_data = output_data.astype(np.float32)

            return torch.from_numpy(output_data)
        
        else:
            raise RuntimeError(f"Acelerador no soportado para ResNet50: {self.target.accelerator}")

    def decode(self, logits: torch.Tensor, top_k: int = 5) -> list[tuple[int, float, str]]:
        print("Decodificando resultados de inferencia")
        
        if self.target.accelerator == Accelerator.CPU or self.target.accelerator == Accelerator.GPU or self.target.accelerator == Accelerator.NPU:
            probabilities = torch.softmax(logits[0], dim=-1)
        elif self.target.accelerator == Accelerator.TPU:
            probabilities = logits[0]
        else:
            raise RuntimeError(f"Acelerador no soportado para ResNet50: {self.target.accelerator}")
        
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
        image_array = np.transpose(image_array, (2, 0, 1)) # Cambiamos de HWC a CHW
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
        self.max_input_chars = 4000
        self.max_output_tokens = 50

    def load(self) -> None:
        print(f"Cargando modelo desde: {self.model_folder_path}")
        if not self.model_folder_path.exists():
            raise FileNotFoundError(f"El modelo no se encontró en la ruta: {self.model_folder_path}")
        
        if self.target.accelerator == Accelerator.CPU or self.target.accelerator == Accelerator.GPU:
            try:
                self.tokenizer = LlamaTokenizerFast.from_pretrained(
                    str(self.model_folder_path),
                    local_files_only=True
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
            
        elif self.target.accelerator == Accelerator.NPU:
            self.model = RKLLM(
                str(self.model_folder_path / "tinyllama.rkllm"),
                max_context_len=self.max_input_chars,
                max_new_tokens=self.max_output_tokens
            )
        
        else:
            raise RuntimeError(f"Acelerador no soportado para TinyLlama: {self.target.accelerator}")
        
        print("Modelo cargado exitosamente")

    def preprocess(self, sample: TextSample) -> dict[str, Any]:
        print(f"Preprocesando muestra: {sample.path}")
        if self.target.accelerator == Accelerator.CPU or self.target.accelerator == Accelerator.GPU:
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
            
        elif self.target.accelerator == Accelerator.NPU:
            prompt_text = sample.prompt[:self.max_input_chars]
            prompt = f"Summarize the following news article:\n\n{prompt_text}"
            return {
                "prompt": prompt
            }
            
        else:
            raise RuntimeError(f"Acelerador no soportado para TinyLlama: {self.target.accelerator}")

    def predict(self, model_inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        print("Ejecutando inferencia en el modelo")
        if self.target.accelerator == Accelerator.CPU or self.target.accelerator == Accelerator.GPU:
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
            
        elif self.target.accelerator == Accelerator.NPU:
            if self.model is None:
                raise RuntimeError("El modelo todavía no está cargado")
            
            global_text.clear()
            self.model.run(model_inputs["prompt"])
            
            return {
                "output_text": "".join(global_text)
            }
            
        else:
            raise RuntimeError(f"Acelerador no soportado para TinyLlama: {self.target.accelerator}")

    def decode(self, logits: dict[str, Any], top_k: int = 5) -> list[dict[str, object]]:
        print("Decodificando resultados de inferencia")
        if self.target.accelerator == Accelerator.CPU or self.target.accelerator == Accelerator.GPU:
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
        
        elif self.target.accelerator == Accelerator.NPU:
            return [
                {
                    "text": logits["output_text"]
                }
            ]
            
        else:
            raise RuntimeError(f"Acelerador no soportado para TinyLlama: {self.target.accelerator}")

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
        self.model = None
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

        self.output_folder_path.mkdir(parents=True, exist_ok=True)
        
        if self.target.accelerator == Accelerator.CPU or self.target.accelerator == Accelerator.GPU:
            self.model = StableDiffusionPipeline.from_pretrained(
                str(self.model_folder_path),
                torch_dtype=torch.float32,
                local_files_only=True,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )

            self.model.to(self.target.device)
            self.model.enable_attention_slicing()
            self.model.set_progress_bar_config(disable=True)
            
        elif self.target.accelerator == Accelerator.NPU:
            scheduler_config_path = str(self.model_folder_path / "scheduler" / "scheduler_config.json")
            with open(scheduler_config_path, "r") as f:
                scheduler_config = json.load(f)
            user_specified_scheduler = LCMScheduler.from_config(scheduler_config)

            self.model = RKNN2LatentConsistencyPipeline(
                text_encoder=RKNN2Model(os.path.join(self.model_folder_path, "text_encoder")),
                unet=RKNN2Model(os.path.join(self.model_folder_path, "unet")),
                vae_decoder=RKNN2Model(os.path.join(self.model_folder_path, "vae_decoder")),
                scheduler=user_specified_scheduler,
                tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16"),
            )
        
        print("Modelo cargado exitosamente")

    def preprocess(self, sample: TextSample) -> dict[str, object]:
        print(f"Preprocesando muestra: {sample.path}")
        if self.model is None:
            raise RuntimeError("El modelo todavía no está cargado")

        if self.target.accelerator == Accelerator.CPU or self.target.accelerator == Accelerator.GPU:
            assert self.target.device is not None
            generator = torch.Generator(device=self.target.device.type).manual_seed(self.seed)
            
            return {
                "sample_path": sample.path,
                "prompt": sample.prompt,
                "generator": generator,
            }
        
        elif self.target.accelerator == Accelerator.NPU:
            generator = np.random.RandomState(self.seed)
            
            return {
                "sample_path": sample.path,
                "prompt": sample.prompt,
                "generator": generator,
            }

    def predict(self, model_inputs: dict[str, object]) -> dict[str, object]:
        print("Ejecutando inferencia en el modelo")
        if self.model is None:
            raise RuntimeError("El modelo todavía no está cargado")

        if self.target.accelerator == Accelerator.CPU or self.target.accelerator == Accelerator.GPU:
            prompt: Any = model_inputs["prompt"]
            generator: Any = model_inputs["generator"]

            result: Any = self.model(
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
        
        elif self.target.accelerator == Accelerator.NPU:
            prompt = model_inputs["prompt"]
            generator = model_inputs["generator"]
            
            result = self.model(
                prompt=prompt,
                height=self.height,
                width=self.width,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
            )
            
            image = result["images"][0]
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
        if self.model is None:
            print("El modelo ya estaba descargado")
            return
        
        if self.target.accelerator == Accelerator.NPU:
            self.model.release()
            

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

        # Para la NPU
        self.labels_path = labels_path
        self.labels = self._load_labels() # La lista de tokens
        self.blank_id = 1024 # El token blank es necesario para la decodificación

    def _resolve_model_path(self) -> Path:
        nemo_files = sorted(self.model_folder_path.glob("*.nemo"))
        if not nemo_files:
            raise FileNotFoundError(
                f"No se encontró ningún modelo en: {self.model_folder_path}"
            )
        return nemo_files[0]

    def _load_labels(self) -> list[str]:
        if self.labels_path is None or not self.labels_path.exists():
            print("Advertencia: No se encontró el archivo de etiquetas (tokens.txt).")
            return []
        
        cleaned_labels = []
        with self.labels_path.open("r", encoding="utf-8") as labels_file:
            for line in labels_file:
                token = line.strip()
                
                if "]" in token and token.startswith("["):
                    token = token.split("]", 1)[1].strip()
                    
                cleaned_labels.append(token)
                
        return cleaned_labels

    def load(self) -> None:
        print(f"Cargando modelo desde: {self.model_folder_path}")
        if not self.model_folder_path.exists():
            raise FileNotFoundError(f"El modelo no se encontró en la ruta: {self.model_folder_path}")
        
        if self.target.accelerator == Accelerator.CPU or self.target.accelerator == Accelerator.GPU:
            # Para evitar que NeMo intente usar GPU cuando se ha especificado CPU, deshabilitamos la visibilidad de las GPUs a nivel de entorno
            if self.target.accelerator == Accelerator.CPU:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
                
            custom_tmp_dir = self.model_folder_path / "nemo_tmp_cache"
            custom_tmp_dir.mkdir(parents=True, exist_ok=True)
            
            os.environ["TMPDIR"] = str(custom_tmp_dir)
            tempfile.tempdir = str(custom_tmp_dir)

            model_path = self._resolve_model_path()
            print("Iniciando restore_from de NeMo...")
            loaded_model: Any = ASRModel.restore_from(
                restore_path=str(model_path),
                map_location=self.target.device,
            )
            self.model = loaded_model
            model: Any = self.model
            model.eval()
            
        elif self.target.accelerator == Accelerator.NPU:
            self.model = [RKNNLite(), RKNNLite()]
            
            # Encoder
            ret = self.model[0].load_rknn(self.model_folder_path / "rnnt_encoder.rknn")
            if ret != 0: raise RuntimeError(f"Error cargando Encoder: {ret}")
            ret = self.model[0].init_runtime()
            if ret != 0: raise RuntimeError(f"Error inicializando Encoder: {ret}")
            
            # Decoder + Joint
            ret = self.model[1].load_rknn(self.model_folder_path / "rnnt_decoder_joint.rknn")
            if ret != 0: raise RuntimeError(f"Error cargando Decoder/Joint: {ret}")
            ret = self.model[1].init_runtime()
            if ret != 0: raise RuntimeError(f"Error inicializando Decoder/Joint: {ret}")
            
        else:
            raise RuntimeError(f"Acelerador no soportado para RNNT: {self.target.accelerator}")
        
        print("Modelo cargado exitosamente")

    def _extract_mel_spectrogram(self, audio_path: str) -> np.ndarray:
        # Cargar audio y asegurar 16kHz mono
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # Parámetros exactos de NeMo Parakeet
        win_length = int(0.025 * 16000) # 400 muestras
        hop_length = int(0.01 * 16000)  # 160 muestras
        n_fft = 512
        
        # Pre-énfasis (0.97)
        padded_wav = torch.nn.functional.pad(waveform, (1, 0))
        preemph_wav = padded_wav[:, 1:] - 0.97 * padded_wav[:, :-1]
        
        # ransformada de Fourier
        window = torch.hann_window(win_length).to(preemph_wav.device)
        stft = torch.stft(
            preemph_wav,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True
        )
        power_spec = torch.abs(stft) ** 2
        
        # Mel Filterbanks (Slaney norm)
        mel_transform = torchaudio.transforms.MelScale(
            n_mels=80,
            sample_rate=16000,
            f_min=0.0,
            f_max=8000.0,
            n_stft=n_fft // 2 + 1,
            norm="slaney",
            mel_scale="slaney"
        ).to(preemph_wav.device)
        
        mel_spec = mel_transform(power_spec)
        logmelspec = torch.log(mel_spec + 1e-5)
        
        # Normalización Per-Feature
        logmelspec_np = logmelspec.numpy().astype(np.float32)
        mean = np.mean(logmelspec_np, axis=2, keepdims=True)
        std = np.std(logmelspec_np, axis=2, keepdims=True)
        
        logmelspec_norm = (logmelspec_np - mean) / (std + 1e-5)
        return logmelspec_norm

    def preprocess(self, sample: AudioSample) -> dict[str, Any]:
        print(f"Preprocesando muestra: {sample.path}")
        if self.model is None:
            raise RuntimeError("El modelo todavía no está cargado")

        if not sample.audio_path.exists():
            raise FileNotFoundError(f"No se encontró el audio en la ruta: {sample.audio_path}")

        # Cargar el audio original
        waveform, sr = torchaudio.load(str(sample.audio_path))
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Recortar exactamente 1 segundo (saltando los primeros 0.4s de silencio)
        # 16000 muestras por segundo. 0.4s = 6400 muestras
        start_sample = int(0.4 * 16000) if waveform.shape[1] > int(1.4 * 16000) else 0
        target_samples = 16000  # 1 segundo exacto
        
        cropped_waveform = waveform[:, start_sample:start_sample + target_samples]
        
        # Rellenar con ceros si el audio original era demasiado corto
        if cropped_waveform.shape[1] < target_samples:
            pad_amount = target_samples - cropped_waveform.shape[1]
            cropped_waveform = torch.nn.functional.pad(cropped_waveform, (0, pad_amount))

        # Guardar en un archivo temporal
        temp_audio_path = Path(tempfile.gettempdir()) / "benchmark_1sec.wav"
        torchaudio.save(str(temp_audio_path), cropped_waveform, 16000)

        model_inputs = {
            "audio_path": str(temp_audio_path),
            "reference": sample.reference,
        }

        if self.target.accelerator == Accelerator.NPU:
            features = self._extract_mel_spectrogram(str(temp_audio_path))
            
            target_frames = 100
            current_frames = features.shape[2]
            
            features = features[:, :, :target_frames]
            current_frames = features.shape[2]
            
            if current_frames < target_frames:
                pad_width = target_frames - current_frames
                features = np.pad(features, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
                
            model_inputs["audio_features"] = features
            model_inputs["audio_length"] = np.array([current_frames], dtype=np.int32)

        return model_inputs

    def _normalize_transcription(self, raw_output: Any) -> str:
        if isinstance(raw_output, list) and raw_output:
            first_output = raw_output[0]
            if isinstance(first_output, str):
                return first_output.strip()
            if hasattr(first_output, "text"):
                return str(first_output.text).strip()

        if isinstance(raw_output, str):
            return raw_output.strip()

        return str(raw_output).strip()

    def predict(self, model_inputs: dict[str, Any]) -> dict[str, Any]:
        print("Ejecutando inferencia en el modelo")
        if self.model is None:
            raise RuntimeError("El modelo todavía no está cargado")

        if self.target.accelerator == Accelerator.CPU or self.target.accelerator == Accelerator.GPU:
            model: Any = self.model
            
            with torch.inference_mode():
                raw_transcription = model.transcribe(
                    audio=[model_inputs["audio_path"]],
                    batch_size=1,
                )
            
            return {
                "text": self._normalize_transcription(raw_transcription),
                "reference": model_inputs.get("reference")
            }

        elif self.target.accelerator == Accelerator.NPU:
            rknn_encoder = self.model[0]
            rknn_decoder = self.model[1]
            
            features = model_inputs["audio_features"]
            length = model_inputs["audio_length"]

            encoder_outputs = rknn_encoder.inference(inputs=[features, length])[0]

            hidden_states_1 = np.zeros((2, 1, 640), dtype=np.float32)
            hidden_states_2 = np.zeros((2, 1, 640), dtype=np.float32)
            
            target = np.array([[self.blank_id]], dtype=np.int64)
            target_length = np.array([1], dtype=np.int64)

            predicted_tokens = []
            T_enc = encoder_outputs.shape[1] if encoder_outputs.shape[-1] == 1024 else encoder_outputs.shape[-1]
            
            max_symbols_per_step = 5

            # Se itera sobre cada milisegundo de audio (cada frame del encoder)
            for t in range(T_enc):
                if encoder_outputs.shape[-1] == 1024:
                    enc_frame = encoder_outputs[:, t:t+1, :].transpose(0, 2, 1)
                else:
                    enc_frame = encoder_outputs[:, :, t:t+1]

                if enc_frame.shape != (1, 1024, 1):
                    enc_frame = enc_frame.reshape(1, 1024, 1)

                symbols_added = 0 

                while True:
                    decoder_outputs = rknn_decoder.inference(
                        inputs=[enc_frame, target, target_length, hidden_states_1, hidden_states_2]
                    )

                    joint_out, new_hidden_1, new_hidden_2 = None, None, None

                    # Identificar los tensores de salida según su tamaño
                    for out_tensor in decoder_outputs:
                        if out_tensor.size == 1280:
                            if new_hidden_1 is None:
                                new_hidden_1 = out_tensor.reshape(2, 1, 640)
                            else:
                                new_hidden_2 = out_tensor.reshape(2, 1, 640)
                        elif out_tensor.size > 1000:
                            joint_out = out_tensor

                    if joint_out is None:
                        raise RuntimeError("No se encontró el tensor de logits en la salida de la NPU.")

                    token = int(np.argmax(joint_out))
                    
                    # Si el token es BLANK, rompemos el bucle y pasamos al siguiente milisegundo de audio
                    if token == self.blank_id or symbols_added >= max_symbols_per_step:
                        break 
                        
                    # Solo si predice una letra real, guardamos y actualizamos la memoria de la red
                    if new_hidden_1 is not None: hidden_states_1 = new_hidden_1
                    if new_hidden_2 is not None: hidden_states_2 = new_hidden_2
                    
                    predicted_tokens.append(token)
                    target = np.array([[token]], dtype=np.int64)
                    symbols_added += 1

            return {
                "predicted_tokens": predicted_tokens,
                "reference": model_inputs.get("reference")
            }

    def decode(self, logits: dict[str, Any], top_k: int = 5) -> list[dict[str, Any]]:
        print("Decodificando resultados de inferencia")

        if self.target.accelerator == Accelerator.CPU or self.target.accelerator == Accelerator.GPU:
            if "text" not in logits:
                raise ValueError("Formato de salida no soportado en RNNT")

            text = str(logits["text"]).strip()
            if not text:
                text = "[sin transcripción]"
            
            return [{ "text": text, "reference": logits.get("reference") }]
            
        elif self.target.accelerator == Accelerator.NPU:
            tokens = logits.get("predicted_tokens", [])
            text_chars = []
            
            for token_id in tokens:
                # Si el modelo predice 0, se salta
                if token_id == self.blank_id: 
                    continue
                    
                # Alinear el ID con el archivo de texto
                if token_id < len(self.labels):
                    char = self.labels[token_id]
                    if char == "<unk>":
                        continue
                    
                    # Reemplazar el marcador especial de espacio
                    char = char.replace("▁", " ")
                    text_chars.append(char)
            
            text = "".join(text_chars).replace("  ", " ").strip()
            if not text:
                text = "[sin transcripción]"
            
            return [{ "text": text, "reference": logits.get("reference") }]
            
        else:
            raise RuntimeError(f"Acelerador no soportado para RNNT: {self.target.accelerator}")

    def infer(self, sample: AudioSample, top_k: int = 5) -> list[dict[str, Any]]:
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
            self.model[0].release()
            self.model[1].release()