import numpy as np

# 1. Definir parámetros
num_muestras = 100  # Entre 100 y 500 es el estándar de la industria para calibrar
mel_features = 80
time_steps = 128

print(f"Generando {num_muestras} muestras de calibración...")

# 2. Generar datos falsos (Dummy Data)
# Usamos una distribución normal estándar (media 0, varianza 1) que simula las características Mel normalizadas
calib_data = np.random.randn(num_muestras, mel_features, time_steps).astype(np.float16)

# 3. Guardar el archivo NumPy
file_name = "calib_audio_signal.npy"
np.save(file_name, calib_data)

print(f"Archivo guardado: {file_name}")
print(f"Forma del tensor guardado: {calib_data.shape} (Muestras, Características, Tiempo)")
print(f"Tipo de dato: {calib_data.dtype}")