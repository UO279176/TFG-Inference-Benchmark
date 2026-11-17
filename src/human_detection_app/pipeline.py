from pathlib import Path
import numpy as np
from PIL import Image

class MobileNetPipeline:
    def __init__(self, model_path: Path, accelerator: str):
        self.model_path = model_path
        self.accelerator = accelerator
        
        self.image_size = 300
        
        self.interpreter = None
        self.input_index = None
        self.input_dtype = None
        self.output_indices = None

        self.tensorflow = None

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo en: {self.model_path}")

        if self.accelerator == "cpu":
            import os
            import tensorflow as tf
            
            with tf.device("/CPU:0"):
                self.interpreter = tf.lite.Interpreter(
                    model_path=str(self.model_path),
                    num_threads=os.cpu_count()
                )
                self.interpreter.allocate_tensors()

            input_details = self.interpreter.get_input_details()[0]
            self.input_index = input_details["index"]
            self.input_dtype = np.dtype(input_details["dtype"])
            self.output_indices = [
                detail["index"] for detail in self.interpreter.get_output_details()
            ]
        
        elif self.accelerator == "gpu":
            import tensorflow as tf
            self.tensorflow = tf

            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                raise RuntimeError("Se pidió GPU pero TensorFlow no detecta ninguna GPU")
            
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            with tf.device("/GPU:0"):
                saved_model = tf.saved_model.load(str(self.model_path))
                self.interpreter = saved_model.signatures["serving_default"]

            input_signature = self.interpreter.structured_input_signature[1]
            _, input_spec = next(iter(input_signature.items()))
            self.input_dtype = np.dtype(input_spec.dtype.as_numpy_dtype)
        
        elif self.accelerator == "tpu":
            from tflite_runtime.interpreter import load_delegate, Interpreter
            from human_detection_app.constants import EDGETPU_LIB_PATH
            
            delegate = load_delegate(EDGETPU_LIB_PATH)
            
            self.interpreter = Interpreter(
                model_path=str(self.model_path),
                experimental_delegates=[delegate]
            )
            self.interpreter.allocate_tensors()
            
            input_details = self.interpreter.get_input_details()[0]
            self.input_index = input_details["index"]
            self.input_dtype = np.dtype(input_details["dtype"])
            self.output_indices = [
                detail["index"] for detail in self.interpreter.get_output_details()
            ]

    def preprocess(self, image_path: Path):
        if self.interpreter is None:
            raise RuntimeError("El modelo todavía no está cargado")

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        
        image_array = np.asarray(image, dtype=self.input_dtype)
        image_array = np.expand_dims(image_array, axis=0) # (1, 300, 300, 3)
        return image_array

    def predict(self, model_inputs):
        if self.interpreter is None:
            raise RuntimeError("El modelo todavía no está cargado")
        
        if self.accelerator == "cpu" or self.accelerator == "tpu":
            self.interpreter.set_tensor(self.input_index, model_inputs)
            self.interpreter.invoke()
            
            results = []
            for output_index in self.output_indices:
                results.append(self.interpreter.get_tensor(output_index))
            return results
        
        elif self.accelerator == "gpu":
            with self.tensorflow.device("/GPU:0"):
                outputs = self.interpreter(
                    inputs=self.tensorflow.constant(model_inputs, dtype=self.input_dtype)
                )
            
            number = int(outputs["num_detections"].numpy())
            locations = outputs["detection_boxes"].numpy()[0][:number]
            categories = outputs["detection_classes"].numpy()[0][:number] - 1
            scores = outputs["detection_scores"].numpy()[0][:number]
            
            return [locations], [categories], [scores], number

    def decode(self, logits):
        if len(logits) != 4:
            raise RuntimeError(f"Se esperaban 4 salidas de detección y se recibieron {len(logits)}")

        locations, categories, scores, number = logits
        return (
            np.asarray(locations)[0],
            np.asarray(categories)[0].astype(np.int32),
            np.asarray(scores)[0],
            int(np.asarray(number).reshape(-1)[0]),
        )

    def infer(self, image_path: Path):
        model_inputs = self.preprocess(image_path)
        logits = self.predict(model_inputs)
        return self.decode(logits)
    
    def unload(self):
        if self.interpreter is not None:
            self.interpreter = None
            self.input_index = None
            self.output_indices = None
            self.input_dtype = None