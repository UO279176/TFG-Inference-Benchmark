import tensorflow as tf
import os

# Cambia esto por el nombre de tu archivo problemático
MODEL_PATH = "encoder_part1_edgetpu_v3.tflite"

if not os.path.exists(MODEL_PATH):
    print(f"No se encuentra el archivo {MODEL_PATH}")
    exit()

print(f"Analizando {MODEL_PATH}...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

ops = interpreter._get_ops_details()
tensors = interpreter.get_tensor_details()

print("\n--- Operaciones ADD encontradas ---")
add_count = 0
for op in ops:
    if op['op_name'] == 'ADD':
        add_count += 1
        # Obtenemos las formas de los dos tensores que se están sumando
        shape_1 = tensors[op['inputs'][0]]['shape']
        shape_2 = tensors[op['inputs'][1]]['shape']
        
        # Solo imprimimos si las formas son diferentes (que es lo que causa el error)
        if list(shape_1) != list(shape_2):
            print(f"ADD #{add_count} -> Input 1: {shape_1} | Input 2: {shape_2}")

print("\nAnálisis terminado")