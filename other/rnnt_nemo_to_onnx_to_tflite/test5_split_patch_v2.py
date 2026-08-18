import onnx

onnx_file = "rnnt_encoder_sim.onnx"
print(f"Cargando {onnx_file}...")
model = onnx.load(onnx_file, load_external_data=True)

splits_fixed = 0

for node in model.graph.node:
    # Se filtran solo los nodos Split del Transformer (los que dividen en Q, K, V)
    if node.op_type == "Split" and len(node.output) == 3:
        axis_found = False
        for attr in node.attribute:
            if attr.name == "axis":
                # Ponemos 1 y el conversor lo transformará en 2
                attr.i = 1  
                axis_found = True
                splits_fixed += 1
        
        if not axis_found:
            new_attr = onnx.helper.make_attribute("axis", 1)
            node.attribute.extend([new_attr])
            splits_fixed += 1

print(f"Nodos Split parcheados: {splits_fixed}")

# Se guarda el nuevo modelo (versión 2)
onnx.save_model(
    model,
    "rnnt_encoder_patched_v2.onnx",
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="rnnt_encoder_patched_v2.onnx.data",
    size_threshold=1024
)
print("Modelo parcheado guardado como 'rnnt_encoder_patched_v2.onnx'")