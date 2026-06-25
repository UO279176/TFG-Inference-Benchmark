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
4. Descargar los modelos, datasets y labels necesarios para la ejecución de los benchmarks y guardarlos en la carpeta "src/data" del proyecto.
5. Construir la imagen de Docker:
```docker build -f <dockerfile a usar> -t tfg-inference-benchmark .```
6. Ejecutar el contenedor de Docker:
- Si se va a usar GPU:
```docker run -it --rm --gpus all -v ./src/data:/app/src/data -v ./results:/app/results tfg-inference-benchmark <acelerador> <modelo>```
- Si se va a usar NPU con RKNN v2.3.0:
```docker run -it --rm --privileged --device=/dev/dri/renderD128:/dev/dri/renderD128 -v ./src/data:/app/src/data -v ./results:/app/results -v ./src/data/libs/librknnrt_230.so:/usr/lib/librknnrt.so -v ./src/data/libs/librkllmrt.so:/usr/lib/librkllmrt.so -v /dev/dri/renderD128:/dev/dri/renderD128 -v /proc/device-tree/compatible:/proc/device-tree/compatible tfg-inference-benchmark <acelerador> <modelo>```
- Si se va a usar NPU con RKNN v2.3.2:
```docker run -it --rm --privileged --device=/dev/dri/renderD128:/dev/dri/renderD128 -v ./src/data:/app/src/data -v ./results:/app/results -v ./src/data/libs/librknnrt_232.so:/usr/lib/librknnrt.so -v ./src/data/libs/librkllmrt.so:/usr/lib/librkllmrt.so -v /dev/dri/renderD128:/dev/dri/renderD128 -v /proc/device-tree/compatible:/proc/device-tree/compatible tfg-inference-benchmark <acelerador> <modelo>```
- Si se va a usar TPU:
```docker run -it --rm --privileged -v ./src/data:/app/src/data -v ./results:/app/results -v /dev/bus/usb:/dev/bus/usb tfg-inference-benchmark <acelerador> <modelo>```

Ejecutar el contenedor sin parámetros mostrará la ayuda con las opciones disponibles.

### Notas adicionales
La carpeta `src/data` no se copia dentro de la imagen para reducir su tamaño. Debe existir en el host y montarse al arrancar el contenedor.
La carpeta `results` también debe montarse si se quiere conservar los CSV generados fuera del contenedor.
Los modelos específicos para un acelerador en concreto se encuentran en la carpeta `src/data/models/<acelerador>` y siguen la misma estructura de carpetas que los modelos generales.