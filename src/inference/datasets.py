from pathlib import Path

from PIL import Image

from inference.contracts import ImageSample


class ImageNetFolderDataset:
    def __init__(self, dataset_folder_path: Path):
        self.dataset_folder_path = dataset_folder_path

    def iter_samples(self, limit: int | None = None) -> list[ImageSample]:
        print(f"Cargando muestras del dataset desde: {self.dataset_folder_path}")
        if not self.dataset_folder_path.exists():
            raise FileNotFoundError(f"No existe el dataset en la ruta: {self.dataset_folder_path}")

        image_files = sorted(self.dataset_folder_path.glob("*.JPEG"))
        if limit is not None:
            image_files = image_files[:limit]

        samples: list[ImageSample] = []
        for image_path in image_files:
            image = Image.open(image_path).convert("RGB")
            samples.append(ImageSample(path=image_path, image=image))

        return samples

class OpenImagesFolderDataset:
    def __init__(self, dataset_folder_path: Path):
        self.dataset_folder_path = dataset_folder_path

    def iter_samples(self, limit: int | None = None) -> list[ImageSample]:
        print(f"Cargando muestras del dataset desde: {self.dataset_folder_path}")
        if not self.dataset_folder_path.exists():
            raise FileNotFoundError(f"No existe el dataset en la ruta: {self.dataset_folder_path}")

        image_files = sorted(self.dataset_folder_path.glob("*.jpg"))
        if limit is not None:
            image_files = image_files[:limit]
        
        samples: list[ImageSample] = []
        for image_path in image_files:
            image = Image.open(image_path).convert("RGB")
            samples.append(ImageSample(path=image_path, image=image))
            
        return samples