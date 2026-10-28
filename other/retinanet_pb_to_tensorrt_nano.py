import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

conversion_params = trt.TrtConversionParams(
    precision_mode=trt.TrtPrecisionMode.FP32,
    max_workspace_size_bytes=1024 * 1024 * 1024
)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='/home/saul/TFG-Inference-Benchmark/src/data/models/retinanet',
    conversion_params=conversion_params
)
converter.convert()

output_dir = '/home/saul/TFG-Inference-Benchmark/model_trt_fp32'
converter.save(output_dir)
print(f"Modelo guardado en {output_dir}")