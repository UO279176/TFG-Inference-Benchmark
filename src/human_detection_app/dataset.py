from pathlib import Path
from PIL import Image

from human_detection_app.dataclasses import HumanImageSample

def load_dataset(dataset_path: Path):
    if not dataset_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {dataset_path}")

    samples = []
    for human_present in [0, 1]: # Dos subcarpetas: 0 (sin humanos) y 1 (con humanos)
        subfolder = dataset_path / str(human_present)
        if not subfolder.exists():
            raise FileNotFoundError(f"No se encontró la subcarpeta: {subfolder}")

        for image_file in subfolder.glob("*.png"):
            samples.append(HumanImageSample(image_file, bool(human_present)))
    
    return samples[920:921] # TODO: Quitar límite