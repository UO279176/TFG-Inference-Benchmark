import sys
from pathlib import Path
import time

from inference.metrics import Metrics
from human_detection_app.pipeline import MobileNetPipeline
from human_detection_app.dataset import load_dataset
from human_detection_app.dataclasses import HumanImageSample, InferenceResult

def get_model_path():
    if accelerator == "cpu":
        model_path = "src/data/models/mobilenet/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
    elif accelerator == "gpu":
        model_path = "src/data/models/mobilenet/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
    elif accelerator == "npu":
        model_path = "src/data/models/mobilenet/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
    elif accelerator == "tpu":
        model_path = "src/data/models/mobilenet/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
    else:
        raise ValueError(f"Tipo de acelerador no válido: {accelerator}")
    return Path(model_path)

# TODO: Comparar resultados con ground truth
def clean_results(result: InferenceResult):
    # Filtrar resultados con categoría igual a 0 (humano) y score mayor o igual a 0.5
    filtered_locations = []
    filtered_categories = []
    filtered_scores = []
    filtered_number = 0

    for loc, cat, score in zip(result.locations, result.categories, result.scores):
        if cat == 0 and score >= 0.5:
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
    print(f"--- Muestra {i + 1} ({sample.path}), presencia humana: {sample.human_present} ---")
    print(f"Número de humanos detectados: {result.number}")
    print(f"Ubicaciones de los humanos detectados: {result.locations}")
    print(f"Categorías de los humanos detectados: {result.categories}")
    print(f"Puntuaciones de los humanos detectados: {result.scores}")
    

if __name__ == "__main__":
    labels_path = Path("src/data/labels/coco-labels-2014_2017.txt")
    dataset_path = Path("src/data/datasets/human_detection")
    accelerator = sys.argv[1]
    model_path = get_model_path()
    samples = load_dataset(dataset_path)
    
    pipeline = MobileNetPipeline(model_path=model_path, labels_path=labels_path, accelerator=accelerator)
    pipeline.load()

    Metrics.start_monitoring(interval_seconds=1.0)
    
    for i, sample in enumerate(samples):
        start_time = time.monotonic()
        inference_result = InferenceResult(*pipeline.infer(sample.path))
        inference_result = clean_results(inference_result)
        show_results(i, sample, inference_result)
        end_time = time.monotonic()
        Metrics.add_inference_time(end_time - start_time) # TODO: Añadir también si acertó la inferencia o falló (es acertado cuando uno de los boxes indica la posición de un humano si el sample contenía un humano o si no se detectaron humanos en un sample donde no hay humanos)
        
    Metrics.stop_monitoring()
    pipeline.unload()
    
    Metrics.print_metrics()
    Metrics.export_metrics_csv(f"results/metrics_mobilenet_{accelerator}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv")