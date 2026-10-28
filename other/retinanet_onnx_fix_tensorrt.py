import onnx

archivo_entrada = "retinanet_v11_clean.onnx"
archivo_salida = "retinanet_def.onnx"

print("Cargando modelo")
modelo = onnx.load(archivo_entrada)

# Parche para Resize
for nodo in modelo.graph.node:
    if nodo.op_type == "Resize":
        for atributo in nodo.attribute:
            if atributo.name == "nearest_mode" and atributo.s == b"floor":
                atributo.s = b"round_prefer_floor"

# Forzar dimensiones estáticas (Batch=1, Alto=800, Ancho=800, Canales=3)
print("Fijando tamaño estático")
for entrada in modelo.graph.input:
    dims = entrada.type.tensor_type.shape.dim
    if len(dims) == 4:
        dims[0].dim_value = 1
        dims[1].dim_value = 800
        dims[2].dim_value = 800
        dims[3].dim_value = 3

onnx.save(modelo, archivo_salida)
print(f"Modelo guardado como: {archivo_salida}")