from inference.contracts import DatasetAdapter, ModelPipeline


class InferenceRunner:
    def __init__(self, model_pipeline: ModelPipeline, dataset_adapter: DatasetAdapter):
        self.model_pipeline = model_pipeline
        self.dataset_adapter = dataset_adapter

    def run_preview(self, max_samples: int = 1, top_k: int = 5) -> None:
        self.model_pipeline.load()
        samples = self.dataset_adapter.iter_samples(limit=max_samples)

        if not samples:
            print("No se encontraron muestras en el dataset")
            return

        for sample in samples:
            predictions = self.model_pipeline.infer(sample=sample, top_k=top_k)
            print(f"\nMuestra: {sample.path.name}")
            if not predictions:
                print("Sin predicciones para esta muestra")
                continue

            for rank, prediction in enumerate(predictions, start=1):
                print(f"{rank}. {self._format_prediction(prediction)}")

    def _format_prediction(self, prediction: object) -> str:
        class_index, label, score = "?", "?", -1.0
        
        if isinstance(prediction, tuple) and len(prediction) == 3:
            class_index, score, label = prediction

        if isinstance(prediction, dict):
            class_index = prediction.get("class_index", prediction.get("class_index", "?"))
            label = prediction.get("label", "?")
            score = prediction.get("score", prediction.get("confidence", -1.0))

        return f"[{class_index}] {label} -> {score:.4f}"
