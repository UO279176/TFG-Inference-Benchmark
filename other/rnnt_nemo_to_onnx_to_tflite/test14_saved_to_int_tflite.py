import tensorflow as tf
import numpy as np

def quantize_for_edgetpu(saved_model_dir, output_path, input_shape):
    print(f"Cuantizando {saved_model_dir}...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    
    # These set the input and output tensors to uint8 (added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    def representative_dataset():
        for _ in range(100):
            yield [np.random.uniform(low=-1.0, high=1.0, size=input_shape).astype(np.float32)]
            
    converter.representative_dataset = representative_dataset
    
    tflite_quant = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_quant)
    print(f"Modelo cuantizado guardado en: {output_path}")

quantize_for_edgetpu("tflite_model_dir_part1_v4", "encoder_part1_static_uint8.tflite_v2", (1, 80, 128))
#quantize_for_edgetpu("tflite_model_dir_part2_v4", "encoder_part2_static_uint8.tflite", (1, 1024, 16))