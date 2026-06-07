from inference.contracts import DatasetAdapter, ModelPipeline
from data import Model, Accelerator
import time

from inference.metrics import Metrics

class InferenceRunner:
    def __init__(self, model_pipeline: ModelPipeline, dataset_adapter: DatasetAdapter, model_identifier: Model, accelerator_identifier: Accelerator):
        self.model_pipeline = model_pipeline
        self.dataset_adapter = dataset_adapter
        self.model_identifier = model_identifier
        self.accelerator_identifier = accelerator_identifier
        
    def run_preview(self, max_samples: int, top_k: int):
        self.model_pipeline.load()
        samples = self.dataset_adapter.iter_samples(limit=max_samples)

        if not samples:
            print("No se encontraron muestras en el dataset")
            return

        Metrics.start_monitoring(interval_seconds=1.0)
        
        for sample in samples:
            start_time = time.monotonic()
            predictions = self.model_pipeline.infer(sample=sample, top_k=top_k)
            end_time = time.monotonic()
            Metrics.add_inference_time(end_time - start_time)
            self.print_inference(sample, predictions)
            
        Metrics.stop_monitoring()
        Metrics.print_metrics()
        Metrics.export_metrics_csv(f"results/metrics_{self.model_identifier.value}_{self.accelerator_identifier.value}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    
    def print_inference(self, sample, predictions):
        print("-" * 50)
        print(f"Muestra: {sample.path.name}")
        for rank, prediction in enumerate(predictions, start=1):
            print(f"{rank}. {self._format_prediction(prediction, self.model_identifier)}")
        print("-" * 50)

    def _format_prediction(self, prediction: object, model_identifier: Model) -> str:
        if model_identifier == Model.RESNET50:
            prediction = prediction if isinstance(prediction, tuple) else (-1, -1.0, "?")
            
            class_index, score, label = prediction
            return f"[{class_index}] {label} -> {score:.4f}"

        if model_identifier == Model.RETINANET:
            prediction = prediction if isinstance(prediction, dict) else {}
            
            class_index = prediction.get("class_index", prediction.get("class_index", "?"))
            label = prediction.get("label", "?")
            score = prediction.get("score", prediction.get("confidence", -1.0))
            return f"[{class_index}] {label} -> {score:.4f}"

        if model_identifier == Model.TINYLLAMA:
            prediction = prediction if isinstance(prediction, dict) else {}
            
            text = prediction.get("text", "?")
            single_line_text = " ".join(text.split())
            return single_line_text

        if model_identifier == Model.STABLE_DIFFUSION:
            prediction = prediction if isinstance(prediction, dict) else {}

            prompt = prediction.get("prompt", "?")
            width = prediction.get("width", "?")
            height = prediction.get("height", "?")
            saved_path = prediction.get("saved_path", "?")
            single_line_prompt = " ".join(str(prompt).split())
            return f"{single_line_prompt} -> {width}x{height} | guardada en: {saved_path}"

        if model_identifier == Model.RNNT:
            prediction = prediction if isinstance(prediction, dict) else {}

            text = str(prediction.get("text", "?")).strip()
            reference = prediction.get("reference", "?").strip()
            return f"pred: {text} | ref: {reference}"

        return "(Formato de predicción desconocido para este modelo)"