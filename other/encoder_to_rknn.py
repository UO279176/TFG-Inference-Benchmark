from rknn.api import RKNN

rknn = RKNN(verbose=True)
rknn.config(target_platform='rk3588', optimization_level=1) 

print("Cargando Encoder ONNX...")
# Define fixed dimensions for standard audio evaluation [Batch, Features, Time]
ret = rknn.load_onnx(
    model='./input/encoder-rnnt_model.onnx',
    inputs=['audio_signal', 'length'],
    input_size_list=[[1, 80, 100], [1]] 
)
if ret != 0:
    print("Error al cargar el modelo.")
    exit(ret)

print("Construyendo RKNN...")
ret = rknn.build(do_quantization=False) # Keep float16 accuracy
if ret != 0:
    print("Error al compilar el modelo")
    exit(ret)

print("Exportando RKNN...")
rknn.export_rknn('./output/encoder.rknn')
rknn.release()
print("Encoder exportado a RKNN")