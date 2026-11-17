from rknn.api import RKNN

PB_MODEL = "frozen_inference_graph.pb"
RKNN_MODEL = "ssd_mobilenet_v2_rk3588_int8.rknn"
CALIB_DATASET = "dataset.txt"

rknn = RKNN(verbose=True)

# mean_values y std_values normalizan los píxeles [0, 255] a [-1, 1] directamente en hardware
print("Configurando modelo para RK3588")
rknn.config(
    mean_values=[[127.5, 127.5, 127.5]],
    std_values=[[127.5, 127.5, 127.5]],
    target_platform="rk3588",
    quantized_dtype="asymmetric_quantized-8"
)

# Cargar grafo FP32 cortando antes el NMS dinámico
print("Cargando grafo")
ret = rknn.load_tensorflow(
    tf_pb=PB_MODEL,
    inputs=["FeatureExtractor/MobilenetV2/MobilenetV2/input"],
    outputs=["concat", "concat_1"],
    input_size_list=[[1, 300, 300, 3]]
)
if ret != 0:
    print("Error cargando el modelo TensorFlow")
    exit(ret)

print("Cuantizando a INT8 y compilando para NPU")
ret = rknn.build(do_quantization=True, dataset=CALIB_DATASET)
if ret != 0:
    print("Error construyendo el modelo RKNN")
    exit(ret)

print("Exportando archivo .rknn")
ret = rknn.export_rknn(RKNN_MODEL)
if ret != 0:
    print("Error exportando el modelo RKNN")
    exit(ret)

rknn.release()
print("Conversión completada, archivo generado:", RKNN_MODEL)