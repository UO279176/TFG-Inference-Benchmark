import sys
from pathlib import Path
import time

from human_detection_app.metrics_with_precision import MetricsP
from human_detection_app.pipeline import MobileNetPipeline
from human_detection_app.dataset import load_dataset
from human_detection_app.dataclasses import HumanImageSample, InferenceResult
from human_detection_app.constants import HUMAN_ID_COCO, HUMAN_DETECTION_DATASET_PATH, get_model_path

def clean_results(result: InferenceResult, score_threshold=0.5):
    # Filtrar resultados para solo tener los correspondientes a humanos y con score mayor o igual a 0.5
    filtered_locations = []
    filtered_categories = []
    filtered_scores = []
    filtered_number = 0

    for loc, cat, score in zip(result.locations, result.categories, result.scores):
        if cat == HUMAN_ID_COCO and score >= score_threshold:
            filtered_locations.append(loc)
            filtered_categories.append(cat)
            filtered_scores.append(score)
            filtered_number += 1
    
    return InferenceResult(
        filtered_locations,
        filtered_categories,
        filtered_scores,
        filtered_number
    )

def show_results(i: int, sample: HumanImageSample, result: InferenceResult):
    print(f"--- Muestra {i + 1} ({sample.path}), humanos reales: {sample.human_count} ---")
    if sample.human_count > 0:
        print(f"Ubicaciones reales de los humanos: {sample.bboxes}")
    print()
    print(f"Número de humanos detectados: {result.number}")
    print(f"Ubicaciones de los humanos detectados: {result.locations}")
    print(f"Puntuaciones de los humanos detectados: {result.scores}")
    print("=" * 50)
    

if __name__ == "__main__":
    dataset_path = Path(HUMAN_DETECTION_DATASET_PATH)
    accelerator = sys.argv[1]
    model_path = get_model_path(accelerator)
    samples = load_dataset(dataset_path)
    
    pipeline = MobileNetPipeline(model_path, accelerator)
    pipeline.load()

    MetricsP.start_monitoring(interval_seconds=1.0)
    
    for i, sample in enumerate(samples):
        start_time = time.monotonic()
        inference_result = InferenceResult(*pipeline.infer(sample.path))
        inference_result = clean_results(inference_result)
        # TODO: Simular un sistema de gestión de salida y almacenamiento de resultados
        end_time = time.monotonic()
        # show_results(i, sample, inference_result)
        MetricsP.add_inference_time(end_time - start_time)
        MetricsP.add_inference_results(sample, inference_result)
        
    MetricsP.stop_monitoring()
    pipeline.unload()
    
    MetricsP.print_metrics()
    MetricsP.export_metrics_csv(f"results/metrics_mobilenet_{accelerator}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv")