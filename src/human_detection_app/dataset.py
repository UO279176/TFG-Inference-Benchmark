import json
from pathlib import Path

from human_detection_app.dataclasses import HumanImageSample
from human_detection_app.constants import HUMAN_DETECTION_DATA_BBOX_JSON

def load_dataset(dataset_path: Path, num_samples=None):
    if not dataset_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {dataset_path}")

    bbox_path = dataset_path / HUMAN_DETECTION_DATA_BBOX_JSON
    if not bbox_path.exists():
        raise FileNotFoundError(f"No se encontró el fichero de bounding boxes: {bbox_path}")

    bbox_annotations = {}
    with bbox_path.open("r", encoding="utf-8") as file:
        for item in json.load(file):
            bbox_annotations[item["file_name"]] = item["bbox"]

    samples = []
    for human_present in [0, 1]: # Dos subcarpetas: 0 (sin humanos) y 1 (con humanos)
        subfolder = dataset_path / str(human_present)
        if not subfolder.exists():
            raise FileNotFoundError(f"No se encontró la subcarpeta: {subfolder}")

        for image_file in sorted(subfolder.glob("*.png")):
            bboxes = bbox_annotations.get(image_file.name, []) if human_present else []
            samples.append(HumanImageSample(image_file, len(bboxes), bboxes))
            if num_samples: # Si se especifica un máximo de muestras, obtener la mitad de cada subcarpeta
                if human_present == 0 and len(samples) >= num_samples // 2:
                    break
                elif human_present == 1 and len(samples) >= num_samples:
                    break
                
    
    return samples