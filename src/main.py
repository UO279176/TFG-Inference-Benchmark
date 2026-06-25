import sys
from dataclasses import replace
from pathlib import Path

import torch

from data import Accelerator, ExecutionTarget, Model, MODEL_RESOURCES
from inference.registry import build_dataset_adapter, build_model_pipeline
from inference.runner import InferenceRunner


accelerators_str = [acc.value for acc in Accelerator]
models_str = [model.value for model in Model]


def resolve_execution_target(accelerator_identifier: Accelerator) -> ExecutionTarget:
    if accelerator_identifier == Accelerator.CPU:
        print("Ejecutando en CPU")
        return ExecutionTarget(accelerator=accelerator_identifier, device=torch.device("cpu"))

    if accelerator_identifier == Accelerator.GPU:
        print("Ejecutando en GPU")
        return ExecutionTarget(accelerator=accelerator_identifier, device=torch.device("cuda"))
    
    if accelerator_identifier == Accelerator.NPU:
        print("Ejecutando en NPU")
        return ExecutionTarget(accelerator=accelerator_identifier, device=None)
    
    if accelerator_identifier == Accelerator.TPU:
        print("Ejecutando en TPU")
        return ExecutionTarget(accelerator=accelerator_identifier, device=None)

    raise NotImplementedError(f"Acelerador no soportado por el runner actual: {accelerator_identifier.value}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in accelerators_str or sys.argv[2] not in models_str:
        print("USO: {0} <{1}> <{2}>".format(sys.argv[0], "|".join(accelerators_str), "|".join(models_str)))
        sys.exit(1)

    accelerator_identifier = Accelerator(sys.argv[1])
    model_identifier = Model(sys.argv[2])
    resources = MODEL_RESOURCES[model_identifier]
    
    # Ajuste de la ruta del modelo para NPU
    if accelerator_identifier == Accelerator.NPU:
        resources = replace(
            resources,
            model_folder_path=resources.model_folder_path.replace("src/data/models", "src/data/models/npu"),
        )
    
    # Ajuste de la ruta del modelo para TPU
    if accelerator_identifier == Accelerator.TPU:
        resources = replace(
            resources,
            model_folder_path=resources.model_folder_path.replace("src/data/models", "src/data/models/tpu"),
        )

    print(f"Nombre: {model_identifier.value}")
    print(f"Ruta modelo: {resources.model_folder_path}")
    print(f"Ruta dataset: {resources.dataset_folder_path}")
    print(f"Ruta labels: {resources.labels_path}")

    try:
        execution_target = resolve_execution_target(accelerator_identifier)
        project_root = Path(__file__).resolve().parent.parent

        model_pipeline = build_model_pipeline(
            model_identifier=model_identifier,
            resources=resources,
            target=execution_target,
            project_root=project_root,
        )
        dataset_adapter = build_dataset_adapter(resources=resources, project_root=project_root)

        runner = InferenceRunner(
            model_pipeline=model_pipeline,
            dataset_adapter=dataset_adapter,
            model_identifier=model_identifier,
            accelerator_identifier=execution_target.accelerator
        )
        runner.run_preview(max_samples=5, top_k=5)

        print("Inferencia completada exitosamente")
    except Exception as e:
        print(f"Error en la ejecución de inferencia: {e}")
        sys.exit(1)