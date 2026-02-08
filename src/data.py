from enum import Enum
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelResources:
    model_path: str
    labels_path: str | None
    dataset_path: str

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

# Asociación de cada modelo con sus recursos
MODEL_RESOURCES = {
    Model.RESNET50: ModelResources(
        model_path="src/data/models/resnet50.h5",
        labels_path="src/data/labels/imagenet_labels.txt",
        dataset_path="src/data/datasets/imagenet_224"
    ),
    Model.RETINANET: ModelResources(
        model_path="",
        labels_path=None,
        dataset_path=""
    ),
    Model.GPT_J: ModelResources(
        model_path="",
        labels_path=None,
        dataset_path=""
    ),
    Model.SDXL: ModelResources(
        model_path="",
        labels_path=None,
        dataset_path=""
    ),
    Model.RNNT: ModelResources(
        model_path="",
        labels_path=None,
        dataset_path=""
    )
}