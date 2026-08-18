import onnx
from onnx import helper

model_path = "rnnt_encoder_fp16_sim.onnx"
print(f"Cargando {model_path}...")
model = onnx.load(model_path)
graph = model.graph

total_nodes = len(graph.node)
mid_index = total_nodes // 2

# 1. Buscar cuellos de botella reales donde solo cruza 1 tensor
print("Analizando topología del grafo en busca de cuellos de botella...")
bottlenecks = []

# Escanear nodos desde el 20% hasta el 80% para evitar los extremos
for split_idx in range(int(total_nodes * 0.2), int(total_nodes * 0.8)):
    part1_nodes = graph.node[:split_idx]
    part2_nodes = graph.node[split_idx:]
    
    p1_outputs = set()
    for n in part1_nodes:
        p1_outputs.update(n.output)
        
    p2_inputs = set()
    for n in part2_nodes:
        p2_inputs.update(n.input)
        
    # Tensores creados en P1 que se usan en P2
    cross_tensors = p1_outputs.intersection(p2_inputs)
    
    if len(cross_tensors) == 1:
        bottlenecks.append((split_idx, list(cross_tensors)[0]))

if not bottlenecks:
    raise ValueError("No se encontraron cuellos de botella limpios en el modelo")

# 2. Elegir el cuello de botella más cercano a la mitad exacta
bottlenecks.sort(key=lambda x: abs(x[0] - mid_index))
best_split_idx, split_tensor_name = bottlenecks[0]

print(f"\nCuello de botella limpio encontrado")
print(f"Cortando en el nodo: {best_split_idx} (Ideal: {mid_index})")
print(f"Tensor puente: '{split_tensor_name}'")

# 3. Dividir el modelo
nodes_part1 = list(graph.node[:best_split_idx])
nodes_part2 = list(graph.node[best_split_idx:])

# Obtener tipo/forma o crear genérico
vi_map = {vi.name: vi for vi in graph.value_info}
split_value_info = vi_map.get(split_tensor_name)
if not split_value_info:
    split_value_info = helper.make_tensor_value_info(
        split_tensor_name, onnx.TensorProto.FLOAT16, [1, -1, 1024]
    )

# --- Guardar Parte 1 ---
p1_inputs = {inp for node in nodes_part1 for inp in node.input}
initializers_part1 = [init for init in graph.initializer if init.name in p1_inputs]

graph_part1 = helper.make_graph(
    nodes=nodes_part1, name="encoder_part1",
    inputs=graph.input, outputs=[split_value_info], initializer=initializers_part1
)
model_part1 = helper.make_model(graph_part1, producer_name="onnx_splitter")
model_part1.opset_import.extend(model.opset_import)
onnx.save(model_part1, "encoder_part1_clean.onnx")
print("'encoder_part1_clean.onnx' guardado")

# --- Guardar Parte 2 ---
p2_inputs = {inp for node in nodes_part2 for inp in node.input}
initializers_part2 = [init for init in graph.initializer if init.name in p2_inputs]

# Asegurar que las entradas globales usadas en la Parte 2 se mantengan
global_inputs_used = [inp for inp in graph.input if inp.name in p2_inputs]

graph_part2 = helper.make_graph(
    nodes=nodes_part2, name="encoder_part2",
    inputs=global_inputs_used + [split_value_info], 
    outputs=graph.output, initializer=initializers_part2
)
model_part2 = helper.make_model(graph_part2, producer_name="onnx_splitter")
model_part2.opset_import.extend(model.opset_import)
onnx.save(model_part2, "encoder_part2_clean.onnx")
print("'encoder_part2_clean.onnx' guardado")