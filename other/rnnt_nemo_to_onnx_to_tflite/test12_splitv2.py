import onnx
from onnx import helper

model_path = "rnnt_encoder_fp16_sim.onnx"
print(f"Cargando {model_path}...")
model = onnx.load(model_path)
graph = model.graph

total_nodes = len(graph.node)
mid_index = total_nodes // 2
print(f"Total de nodos en el grafo: {total_nodes}")
print(f"Punto de corte elegido (índice exacto): {mid_index}")

# 1. Dividir los nodos en dos mitades
nodes_part1 = list(graph.node[:mid_index])
nodes_part2 = list(graph.node[mid_index:])

# 2. Encontrar todos los tensores producidos en la Parte 1
part1_outputs_set = set()
for node in nodes_part1:
  for out in node.output:
    part1_outputs_set.add(out)

# 3. Detectar todos los tensores que cruzan de la Parte 1 a la Parte 2
cross_tensors = set()
for node in nodes_part2:
  for inp in node.input:
    if inp in part1_outputs_set:
      cross_tensors.add(inp)

print(f"Se han detectado {len(cross_tensors)} tensores cruzando la frontera de corte")

# 4. Obtener información de tipos y formas de los tensores existentes
vi_map = {vi.name: vi for vi in graph.value_info}
for inp in graph.input:
  vi_map[inp.name] = inp

# Crear value_info para los tensores de cruce que no estén explícitamente registrados
cross_value_infos = []
for t_name in cross_tensors:
  if t_name in vi_map:
    cross_value_infos.append(vi_map[t_name])
  else:
    cross_value_infos.append(
        helper.make_tensor_value_info(
            t_name, onnx.TensorProto.FLOAT16, [1, -1, 1024]
        )
    )

# --- Construir Parte 1 ---
p1_inputs = {inp for node in nodes_part1 for inp in node.input}
initializers_part1 = [
    init for init in graph.initializer if init.name in p1_inputs
]

graph_part1 = helper.make_graph(
    nodes=nodes_part1,
    name="encoder_part1_v2",
    inputs=graph.input,
    outputs=cross_value_infos, # Todas las conexiones de cruce son salidas ahora
    initializer=initializers_part1,
)
model_part1 = helper.make_model(graph_part1, producer_name="onnx_splitter")

del model_part1.opset_import[:]
for op in model.opset_import:
  new_op = model_part1.opset_import.add()
  new_op.CopyFrom(op)

onnx.save(model_part1, "encoder_part1_v2.onnx")
print("'encoder_part1_v2.onnx' guardado")

# --- Construir Parte 2 ---
p2_inputs = list(graph.input) + cross_value_infos
p2_node_inputs = {inp for node in nodes_part2 for inp in node.input}
initializers_part2 = [
    init for init in graph.initializer if init.name in p2_node_inputs
]

graph_part2 = helper.make_graph(
    nodes=nodes_part2,
    name="encoder_part2_v2",
    inputs=p2_inputs, # Las conexiones de cruce entran como inputs aquí
    outputs=graph.output,
    initializer=initializers_part2,
)
model_part2 = helper.make_model(graph_part2, producer_name="onnx_splitter")

del model_part2.opset_import[:]
for op in model.opset_import:
  new_op = model_part2.opset_import.add()
  new_op.CopyFrom(op)

onnx.save(model_part2, "encoder_part2_v2.onnx")
print("'encoder_part2_v2.onnx' guardado")