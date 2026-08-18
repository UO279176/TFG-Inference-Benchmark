import tensorflow as tf
import numpy as np

def quantize_for_edgetpu(saved_model_dir, output_path):
    print(f"\n--- Cuantizando {saved_model_dir} ---")
    
    # 1. Leer la forma que onnx2tf ha decidido usar
    temp_converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    temp_tflite = temp_converter.convert()
    interpreter = tf.lite.Interpreter(model_content=temp_tflite)
    input_details = interpreter.get_input_details()[0]
    real_shape = input_details['shape']
    print(f"Forma de entrada detectada: {real_shape}")

    # 2. Configurar el conversor real para INT8
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # 3. Generar datos
    def representative_dataset():
        for _ in range(100):
            yield [np.random.uniform(low=-1.0, high=1.0, size=real_shape).astype(np.float32)]
            
    converter.representative_dataset = representative_dataset
    
    tflite_quant = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_quant)
    print(f"Modelo guardado en: {output_path}")

quantize_for_edgetpu("tflite_model_dir_part1_v5", "encoder_part1_edgetpu_v3.tflite")
#quantize_for_edgetpu("tflite_part2_fp32", "encoder_part2_edgetpu.tflite")