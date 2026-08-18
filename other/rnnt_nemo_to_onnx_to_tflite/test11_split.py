import onnx
from onnx import helper

# 1. Cargar el modelo
model_path = "rnnt_encoder_fp16_sim.onnx"
print(f"Cargando {model_path}...")
model = onnx.load(model_path)
graph = model.graph

total_nodes = len(graph.node)
mid_index = total_nodes // 2

print(f"Total de nodos en el grafo: {total_nodes}")
print(f"Punto de corte elegido (índice exacto): {mid_index}")

# 2. Seleccionar el tensor de puente (la salida del nodo de corte)
bridge_node = graph.node[mid_index - 1]
if not bridge_node.output:
  raise ValueError(
      "El nodo de corte no tiene salidas definidas. Ajusta el índice mid_index."
  )

split_tensor_name = bridge_node.output[0]
print(f"Tensor de puente seleccionado: '{split_tensor_name}'")

# 3. Dividir los nodos en dos listas
nodes_part1 = list(graph.node[:mid_index])
nodes_part2 = list(graph.node[mid_index:])

# 4. Buscar información del tensor de puente en value_info (o crearla)
split_value_info = None
for vi in graph.value_info:
  if vi.name == split_tensor_name:
    split_value_info = vi
    break

if split_value_info is None:
  split_value_info = helper.make_tensor_value_info(
      split_tensor_name, onnx.TensorProto.FLOAT16, [1, -1, 512]
  )

# --- Construir Parte 1 ---
p1_inputs = {inp for node in nodes_part1 for inp in node.input}
initializers_part1 = [
    init for init in graph.initializer if init.name in p1_inputs
]

graph_part1 = helper.make_graph(
    nodes=nodes_part1,
    name="encoder_part1",
    inputs=graph.input,
    outputs=[split_value_info],
    initializer=initializers_part1,
)
model_part1 = helper.make_model(graph_part1, producer_name="onnx_splitter")

# Corrección de opset usando bucle seguro
del model_part1.opset_import[:]
for op in model.opset_import:
  new_opset = model_part1.opset_import.add()
  new_opset.CopyFrom(op)

onnx.save(model_part1, "encoder_part1.onnx")
print("'encoder_part1.onnx' guardado correctamente")

# --- Construir Parte 2 ---
p2_inputs = {inp for node in nodes_part2 for inp in node.input}
initializers_part2 = [
    init for init in graph.initializer if init.name in p2_inputs
]

graph_part2 = helper.make_graph(
    nodes=nodes_part2,
    name="encoder_part2",
    inputs=[split_value_info],
    outputs=graph.output,
    initializer=initializers_part2,
)
model_part2 = helper.make_model(graph_part2, producer_name="onnx_splitter")

# Corrección de opset usando bucle seguro
del model_part2.opset_import[:]
for op in model.opset_import:
  new_opset = model_part2.opset_import.add()
  new_opset.CopyFrom(op)

onnx.save(model_part2, "encoder_part2.onnx")
print("'encoder_part2.onnx' guardado correctamente")