import sys
from data import Accelerator, Model, MODEL_RESOURCES
from pathlib import Path
from transformers import AutoModel
import torch

accelerators_str = [acc.value for acc in Accelerator]
models_str = [model.value for model in Model]

def loadModel(model_relative_path: str, device: torch.device):
    try:
        model_absolute_path = Path(__file__).resolve().parent.parent / model_relative_path
        if not model_absolute_path.exists():
            raise FileNotFoundError(f"El modelo no se encontró en la ruta: {model_absolute_path}")
        
        model = AutoModel.from_pretrained(str(model_absolute_path))
        model.to(device)
        print("Modelo cargado exitosamente")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in accelerators_str or sys.argv[2] not in models_str:
        print("USO: {0} <{1}> <{2}>".format(sys.argv[0], "|".join(accelerators_str), "|".join(models_str)))
        sys.exit(1)
    
    accelerator_identifier = Accelerator(sys.argv[1])
    model_identifier = Model(sys.argv[2])
    resources = MODEL_RESOURCES[model_identifier]

    print(f"Nombre: {model_identifier.value}")
    print(f"Ruta modelo: {resources.model_folder_path}")
    print(f"Ruta dataset: {resources.dataset_path}")
    print(f"Ruta labels: {resources.labels_path}")
    
    if accelerator_identifier == Accelerator.CPU:
        print("Ejecutando en CPU")
        device = torch.device("cpu")
        new_model = loadModel(resources.model_folder_path, device)
    elif accelerator_identifier == Accelerator.GPU:
        print("Ejecutando en GPU")
        if not torch.cuda.is_available():
            print("No se encontró una GPU con CUDA disponible")
            sys.exit(1)
            
        device = torch.device("cuda")
        new_model = loadModel(resources.model_folder_path, device)