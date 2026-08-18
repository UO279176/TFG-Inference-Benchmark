import onnx

onnx_file = "rnnt_encoder_sim.onnx"
print(f"Cargando {onnx_file}...")
model = onnx.load(onnx_file, load_external_data=True)

splits_fixed = 0

# Se itera sobre la estructura interna del modelo
for node in model.graph.node:
    if node.op_type == "Split":
        # Las proyecciones QKV de los Transformers siempre tienen 3 salidas
        if len(node.output) == 3:
            axis_found = False
            for attr in node.attribute:
                if attr.name == "axis":
                    # Forzamos la división en la última dimensión (-1)
                    attr.i = -1  
                    axis_found = True
                    splits_fixed += 1
            
            # Si ONNX omitió el atributo (usa default 0), lo inyectamos manualmente
            if not axis_found:
                new_attr = onnx.helper.make_attribute("axis", -1)
                node.attribute.extend([new_attr])
                splits_fixed += 1

print(f"Nodos Split parcheados: {splits_fixed}")

# Se guarda el nuevo modelo parcheado
onnx.save_model(
    model,
    "rnnt_encoder_patched.onnx",
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="rnnt_encoder_patched.onnx.data",
    size_threshold=1024
)
print("Modelo parcheado guardado como 'rnnt_encoder_patched.onnx'")