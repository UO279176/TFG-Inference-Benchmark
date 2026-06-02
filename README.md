# TFG-Inference-Benchmark
Repositorio para mi TFG: Comparativa de dispositivos SBC de bajo coste para la ejecución de modelos de inferencia de IA.

En esto repositorio se encuentra el código fuente del proyecto. El objetivo del proyecto es evaluar el rendimiento de diferentes dispositivos SBC (Single Board Computers) de bajo coste en la ejecución de modelos de inferencia de IA.

## Preparación del entorno y ejecución
Para preparar el entorno de ejecución se utilizará Docker y se deben seguir los siguientes pasos:
1. Asegurarse de tener Docker instalado y en funcionamiento.
2. Clonar el repositorio:
```git clone https://github.com/UO279176/TFG-Inference-Benchmark.git```
3. Acceder al directorio del proyecto:
```cd TFG-Inference-Benchmark```
4. Crear un archivo .env con una variable de entorno que indique la antigüedad de la arquitectura CUDA a utilizar ("old" para CUDA 10.2 y "new" para CUDA 12.6 o superior):
```CUDA_TARGET=<new|old>```
5. Descargar los modelos, datasets y labels necesarios para la ejecución de los benchmarks y guardarlos en la carpeta "data" del proyecto.
6. Construir la imagen de Docker:
```docker build -t tfg-inference-benchmark .```
7. Ejecutar el contenedor de Docker:
```docker run -it --rm --gpus all tfg-inference-benchmark <acelerador> <modelo>```
Ejecutar el contenedor sin parámetros mostrará la ayuda con las opciones disponibles.