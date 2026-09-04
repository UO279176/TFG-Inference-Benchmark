import numpy as np
from ai_edge_quantizer import quantizer, recipe

def generate_calibration_data(num_samples=100):
    samples = []
    for _ in range(num_samples):
        # 1. args_0 -> Equivalente a los 'input_ids' (tokens)
        # Generamos tokens ficticios (del 0 al 32000), shape (1, 128), tipo int64
        sample_args_0 = np.random.randint(0, 32000, size=(1, 128)).astype(np.int64)
        
        # 2. args_1 -> Equivalente a la 'attention_mask'
        # Para la calibración simulada, podemos asumir que todos los tokens son válidos (todo unos).
        # shape (1, 128), tipo int64
        sample_args_1 = np.ones((1, 128), dtype=np.int64)
        
        # 3. Empaquetamos AMBAS entradas tal y como exige el modelo
        sample_dict = {
            "args_0": sample_args_0,
            "args_1": sample_args_1
        } 
        
        samples.append(sample_dict)
    return samples

calib_samples = generate_calibration_data()

qt = quantizer.Quantizer("tinyllama.tflite")
qt.load_quantization_recipe(recipe.static_wi8_ai8())
calibration_result = qt.calibrate({"serving_default": calib_samples})
qt.quantize(calibration_result).export_model("tinyllama_quantized.tflite")