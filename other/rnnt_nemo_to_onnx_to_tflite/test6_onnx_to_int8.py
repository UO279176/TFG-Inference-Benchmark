import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

onnx_in = "rnnt_encoder_patched_v2.onnx"
onnx_out = "rnnt_encoder_int8.onnx"

print("Iniciando cuantización dinámica a INT8...")

quantize_dynamic(
    model_input=onnx_in,
    model_output=onnx_out,
    weight_type=QuantType.QUInt8,
    use_external_data_format=True
)

print(f"Modelo cuantizado como: {onnx_out}")