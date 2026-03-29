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
            print(f"\nImagen: {sample.path.name}")
            for rank, (class_index, score, label) in enumerate(predictions, start=1):
                print(f"{rank}. [{class_index}] {label} -> {score:.4f}")
