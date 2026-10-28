# Script para convertir Resnet50 a pytorch_model.bin, para la NVIDIA Jetson Nano y su increíblemente limitado Python 3.6

import torch
from safetensors.torch import load_file

path_modelo = "src/data/models/resnet50"
state_dict = load_file(f"{path_modelo}/model.safetensors")

# Formato PyTorch binario clásico que entiende Python 3.6
torch.save(state_dict, f"{path_modelo}/pytorch_model.bin")

print("Creado pytorch_model.bin correctamente")