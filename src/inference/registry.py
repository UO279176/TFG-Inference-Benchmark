from pathlib import Path
from typing import Callable

import torch

from data import Dataset, Model, ModelResources
from inference.contracts import DatasetAdapter, ModelPipeline
from inference.datasets import ImageNetFolderDataset, OpenImagesFolderDataset
from inference.pipelines import ResNet50Pipeline, RetinaNetPipeline

MODEL_PIPELINE_REGISTRY: dict[Model, Callable[..., ModelPipeline]] = {
    Model.RESNET50: ResNet50Pipeline,
    Model.RETINANET: RetinaNetPipeline
}

DATASET_ADAPTER_REGISTRY: dict[Dataset, Callable[..., DatasetAdapter]] = {
    Dataset.IMAGENET: ImageNetFolderDataset,
    Dataset.OPENIMAGES: OpenImagesFolderDataset
}


def build_model_pipeline(
    model_identifier: Model,
    resources: ModelResources,
    device: torch.device,
    project_root: Path,
) -> ModelPipeline:
    pipeline_class = MODEL_PIPELINE_REGISTRY.get(model_identifier)
    if pipeline_class is None:
        raise NotImplementedError(f"No hay pipeline implementado para el modelo: {model_identifier.value}")

    model_folder_path = project_root / resources.model_folder_path
    labels_path = project_root / resources.labels_path if resources.labels_path else None
    
    return pipeline_class(
        model_folder_path=model_folder_path,
        labels_path=labels_path,
        device=device
    )


def build_dataset_adapter(resources: ModelResources, project_root: Path) -> DatasetAdapter:
    if resources.dataset is None:
        raise NotImplementedError("El modelo no tiene dataset asociado")

    dataset_class = DATASET_ADAPTER_REGISTRY.get(resources.dataset)
    if dataset_class is None:
        raise NotImplementedError(f"No hay adapter implementado para dataset: {resources.dataset.value}")

    dataset_folder_path = project_root / resources.dataset_folder_path
    return dataset_class(dataset_folder_path=dataset_folder_path)
