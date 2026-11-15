import sys
from pathlib import Path
import time
import json

from human_detection_app.metrics_with_precision import MetricsP
from human_detection_app.pipeline import MobileNetPipeline
from human_detection_app.dataset import load_dataset
from human_detection_app.dataclasses import HumanImageSample, InferenceResult
from human_detection_app.constants import HUMAN_ID_COCO, HUMAN_DETECTION_DATASET_PATH, get_model_path

def build_result(sample: HumanImageSample, result: InferenceResult):
    # Construir un paquete de resultados para enviar a un sistema de gestión y almacenado de resultados
    if result.number == 0:
        return None
    
    event_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame": sample.path.name,
        "human_count": result.number,
        "detections": [
            {
                "bbox": [float(coord) for coord in loc.tolist()],
                "score": float(score)
            }
            for loc, score in zip(result.locations, result.scores)
        ]
    }
    return json.dumps(event_payload)

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
        print(f"Muestra {i + 1} de {len(samples)}...", end="\r")
        start_time = time.monotonic()
        inference_result = InferenceResult(*pipeline.infer(sample.path)) # Carga de imágenes + inferencia
        inference_result = clean_results(inference_result) # Filtrado de resultados
        payload = build_result(sample, inference_result) # Empaquetado de resultados
        end_time = time.monotonic()
        # show_results(i, sample, inference_result)
        MetricsP.add_inference_time(end_time - start_time)
        MetricsP.add_inference_results(sample, inference_result)
        
    MetricsP.stop_monitoring()
    pipeline.unload()
    
    MetricsP.print_metrics()
    MetricsP.export_metrics_csv(f"results/metrics_mobilenet_{accelerator}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv")