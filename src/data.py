from __future__ import annotations

from enum import Enum
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelResources:
    model_folder_path: str
    labels_path: str | None
    dataset: Dataset | None
    dataset_folder_path: str

class Accelerator(Enum):
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    TPU = "tpu"
    
# Nombre de los modelos. Estos serán usados para validar la entrada del usuario
class Model(Enum):
    RESNET50 = "resnet50"
    RETINANET = "retinanet"
    GPT_J = "gpt-j"
    SDXL = "sdxl"
    RNNT = "rnnt"

class Dataset(Enum):
    IMAGENET = "imagenet"

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
        labels_path=None,
        dataset=None,
        dataset_folder_path=""
    ),
    Model.GPT_J: ModelResources(
        model_folder_path="",
        labels_path=None,
        dataset=None,
        dataset_folder_path=""
    ),
    Model.SDXL: ModelResources(
        model_folder_path="",
        labels_path=None,
        dataset=None,
        dataset_folder_path=""
    ),
    Model.RNNT: ModelResources(
        model_folder_path="",
        labels_path=None,
        dataset=None,
        dataset_folder_path=""
    )
}