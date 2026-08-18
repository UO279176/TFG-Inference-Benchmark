import onnx
from onnxsim import simplify

onnx_in = "rnnt_encoder_fp16.onnx"
onnx_out = "rnnt_encoder_fp16_sim.onnx"

print("1. Cargando el modelo")
# load_external_data=True permite cargar modelos divididos
model = onnx.load(onnx_in, load_external_data=True)

print("2. Simplificando el grafo")
# Simplificamos el grafo
model_simp, check = simplify(model)

if not check:
    print("Advertencia: La validación del modelo simplificado falló, pero se intentará guardar igualmente")

print("3. Guardando el modelo simplificado")
# Guardamos separando la estructura (.onnx) de los pesos (.data)
onnx.save_model(
    model_simp,
    onnx_out,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="rnnt_encoder_fp16_sim.onnx.data",
    size_threshold=1024  # Extrae cualquier tensor mayor a 1KB
)

print(f"Modelo simplificado guardado en: {onnx_out}")