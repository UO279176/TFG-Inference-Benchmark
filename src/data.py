from __future__ import annotations

from enum import Enum
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelResources:
    model_folder_path: str # Ruta relativa del modelo
    labels_path: str | None # Ruta relativa del archivo de etiquetas (si aplica)
    dataset: Dataset | None # Tipo de dataset asociado al modelo
    dataset_folder_path: str # Ruta relativa del dataset asociado al modelo

class Accelerator(Enum):
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    TPU = "tpu"
    
# Nombre de los modelos. Estos serán usados para validar la entrada del usuario
class Model(Enum):
    RESNET50 = "resnet50"
    RETINANET = "retinanet"
    TINYLLAMA = "tinyllama"
    STABLE_DIFFUSION = "stable_diffusion"
    RNNT = "rnnt"

class Dataset(Enum):
    IMAGENET = "imagenet"
    OPENIMAGES = "openimages"
    CNN_DAILYMAIL_NEWS = "cnn_dailymail_news"
    LIBRISPEECH = "librispeech"
    STABLE_DIFFUSION_PROMPTS = "stable_diffusion_prompts"

# Asociación de cada modelo con sus recursos
MODEL_RESOURCES = {
    Model.RESNET50: ModelResources(
        model_folder_path="src/data/models/resnet50",
        labels_path="src/data/labels/imagenet_labels.txt",
        dataset=Dataset.IMAGENET,
        dataset_folder_path="src/data/datasets/imagenet"
    ),
    Model.RETINANET: ModelResources(
        model_folder_path="src/data/models/retinanet",
        labels_path="src/data/labels/coco-labels-2014_2017.txt",
        dataset=Dataset.OPENIMAGES,
        dataset_folder_path="src/data/datasets/openimages"
    ),
    Model.TINYLLAMA: ModelResources(
        model_folder_path="src/data/models/tinyllama",
        labels_path=None,
        dataset=Dataset.CNN_DAILYMAIL_NEWS,
        dataset_folder_path="src/data/datasets/cnn_dailymail_news"
    ),
    Model.STABLE_DIFFUSION: ModelResources(
        model_folder_path="src/data/models/stablediffusion15",
        labels_path=None,
        dataset=Dataset.STABLE_DIFFUSION_PROMPTS,
        dataset_folder_path="src/data/datasets/stable_diffusion_prompts"
    ),
    Model.RNNT: ModelResources(
        model_folder_path="",
        labels_path=None,
        dataset=None,
        dataset_folder_path=""
    )
}