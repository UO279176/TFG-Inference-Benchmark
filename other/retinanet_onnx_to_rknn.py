from rknn.api import RKNN

rknn = RKNN(verbose=True)

print('Configurando el modelo...')
rknn.config(
    mean_values=[[123.675, 116.28, 103.53]],  
    std_values=[[58.395, 57.12, 57.375]],     
    target_platform='rk3588'
)

print('Cargando el modelo ONNX unificado...')
ret = rknn.load_onnx(
    model='/workspace/input_models/retinanet.onnx',
    inputs=['input_1:0'],
    # Dimensiones alineadas con el optimizador de ONNXRuntime
    input_size_list=[[1, 3, 800, 800]] 
)

if ret != 0:
    print('Error: La carga del modelo falló.')
    exit(ret)

print('Construyendo el modelo RKNN...')
ret = rknn.build(do_quantization=False)
if ret != 0:
    print('Error: La construcción del modelo falló.')
    exit(ret)

print('Exportando a formato .rknn...')
ret = rknn.export_rknn('/workspace/input_models/retinanet.rknn')
if ret != 0:
    print('Error: La exportación del modelo falló.')
    exit(ret)

print('Modelo compilado en .rknn')
rknn.release()