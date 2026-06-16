import numpy as np
from rknn.api import RKNN

rknn = RKNN(verbose=True)
rknn.config(
    mean_values=[],
    std_values=[],
    target_platform='rk3588',
    optimization_level=1
)

# Cargar el archivo ONNX
print("Cargando el ONNX Decoder + Joint...")
ret = rknn.load_onnx(
    model='./input/decoder_joint-rnnt_model.onnx',
    inputs=['encoder_outputs', 'targets', 'target_length', 'input_states_1', 'input_states_2'],
    input_size_list=[[1, 1024, 1], [1, 1], [1], [2, 1, 640], [2, 1, 640]]
)
if ret != 0:
    print("Error al cargar el archivo ONNX.")
    exit(ret)

# Compilar el modelo RKNN
print("Compilando el modelo a RKNN...")
ret = rknn.build(do_quantization=False) 
if ret != 0:
    print("Error durante la compilación del modelo.")
    exit(ret)

# Exportar el binario final
output_path = './output/decoder_joint.rknn'
ret = rknn.export_rknn(output_path)
if ret != 0:
    print("Error al exportar el archivo .rknn final.")
    exit(ret)

print(f"Modelo combinado guardado en: {output_path}")
rknn.release()