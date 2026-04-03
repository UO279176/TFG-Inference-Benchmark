from inference.contracts import DatasetAdapter, ModelPipeline
from data import Model

class InferenceRunner:
    def __init__(self, model_pipeline: ModelPipeline, dataset_adapter: DatasetAdapter, model_identifier: Model):
        self.model_pipeline = model_pipeline
        self.dataset_adapter = dataset_adapter
        self.model_identifier = model_identifier

    def run_preview(self, max_samples: int = 1, top_k: int = 5) -> None:
        self.model_pipeline.load()
        samples = self.dataset_adapter.iter_samples(limit=max_samples)

        if not samples:
            print("No se encontraron muestras en el dataset")
            return

        for sample in samples:
            predictions = self.model_pipeline.infer(sample=sample, top_k=top_k)
            print("-" * 50)
            print(f"Muestra: {sample.path.name}")
            if not predictions:
                print("Sin predicciones para esta muestra")
                continue

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

        return "(Formato de predicción desconocido para este modelo)"