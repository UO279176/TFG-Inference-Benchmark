# TFG-Inference-Benchmark
Repositorio para mi TFG: Comparativa de dispositivos SBC de bajo coste para la ejecución de modelos de inferencia de IA.

En esto repositorio se encuentra el código fuente del proyecto. El objetivo del proyecto es evaluar el rendimiento de diferentes dispositivos SBC (Single Board Computers) de bajo coste en la ejecución de modelos de inferencia de IA.

Para ejecutar el programa, usar el comando:
```python3 src/main.py <acelerador> <modelo> <num_muestras> [-v]```<br/>
- `<acelerador>`: Identificador del acelerador a usar (CPU, GPU, NPU, TPU).<br/>
- `<modelo>`: Identificador del modelo a usar (Comprobar strings de la clase Model en src/data.py).<br/>
-`<num_muestras>`: Número de muestras a procesar del dataset.<br/>
- `-v | --verbose`: (Opcional) Modo verbose, muestra información detallada de cada muestra procesada, como resultados de predicción y por qué paso del pipeline se encuentra.

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
```docker run -it --rm -v ./src/data:/app/src/data -v ./results:/app/results tfg-inference-benchmark <acelerador> <modelo> <num_muestras>```
- Si se va a usar GPU, añadir estos parámetros al comando de ejecución:
```-gpus all```
- Si se va a usar NPU con RKNN v2.3.0, añadir estos parámetros al comando de ejecución:
```--privileged --device=/dev/dri/renderD128:/dev/dri/renderD128 -v ./src/data/libs/librknnrt_230.so:/usr/lib/librknnrt.so -v ./src/data/libs/librkllmrt.so:/usr/lib/librkllmrt.so -v /dev/dri/renderD128:/dev/dri/renderD128 -v /proc/device-tree/compatible:/proc/device-tree/compatible```
- Si se va a usar NPU con RKNN v2.3.2, añadir estos parámetros al comando de ejecución:
```--privileged --device=/dev/dri/renderD128:/dev/dri/renderD128 -v ./src/data/libs/librknnrt_232.so:/usr/lib/librknnrt.so -v ./src/data/libs/librkllmrt.so:/usr/lib/librkllmrt.so -v /dev/dri/renderD128:/dev/dri/renderD128 -v /proc/device-tree/compatible:/proc/device-tree/compatible```
- Si se va a usar TPU, añadir estos parámetros al comando de ejecución:
```--privileged -v /dev/bus/usb:/dev/bus/usb```

Ejecutar el contenedor sin parámetros mostrará la ayuda con las opciones disponibles.

## Alternativa manual sin Docker
En caso de no querer usar Docker, ya sea por comodidad o porque el uso de Docker ha dado problemas en algún paso, se puede instalar y ejecutar manualmente el entorno:

- Para la GPU de la Jetson AGX Orin y Jetson Orin Nano:
1. Crear un entorno virtual de Python:<br/>
```python3 -m venv venv```
2. Activar el entorno virtual:<br/>
```source venv/bin/activate```
3. Instalar las dependencias en el siguiente orden:<br/>
```pip3 install torch torchaudio cuda-python --index-url https://pypi.jetson-ai-lab.io/jp6/cu126```<br/>
```pip3 install nvidia-cudss-cu12```<br/>
```pip3 install numpy```<br/>
```pip3 install tensorflow==2.16.1+nv24.08 --index-url https://developer.nvidia.com/w/w/compute/redist/jp/v61 --no-deps```<br/>
```pip3 install gast astunparse opt_einsum flatbuffers```<br/>
```pip3 install -r requirements-cuda-new.txt```
4. Desinstalar cublas para que no haya conflictos con la versión de cublas instalada en el sistema:<br/>
```pip3 uninstall nvidia-cublas-cu12```
5. Configurar la variable de entorno LD_LIBRARY_PATH, sustituyendo `<ruta_venv>` por la ruta absoluta del entorno virtual creado (obligatorio siempre que se active el entorno virtual):<br/>
```export LD_LIBRARY_PATH="<ruta_venv>/lib/python3.10/site-packages/nvidia/cu12/lib/:$LD_LIBRARY_PATH"```
6. Descargar las librerías de FFMPEG versión 7.1.5. En este caso, no se tiene acceso privilegiado al sistema, por lo que se descargará y compilará de forma local:<br/>
```wget https://ffmpeg.org/releases/ffmpeg-7.1.5.tar.bz2```<br/>
```tar -xf ffmpeg-7.1.5.tar.bz2 && cd ffmpeg-7.1.5```<br/>
```./configure --prefix=$HOME/ffmpeg_local_build --enable-shared --disable-static --disable-programs --disable-doc```<br/>
```make -j$(nproc) && make install```
7. Configurar la variable de entorno LD_LIBRARY_PATH para que incluya la ruta de las librerías de FFMPEG compiladas (obligatorio siempre que se active el entorno virtual):<br/>
```export LD_LIBRARY_PATH="$HOME/ffmpeg_local_build/lib/:$LD_LIBRARY_PATH"```
8. Compilar e instalar torchcodec, ya que la versión precompilada en el repositorio usado para instalar las dependencias no es compatible con ese PyTorch precompilado:<br/>
```pip3 uninstall -y torchcodec``` (en caso de haberlo instalado anteriormente)<br/>
```pip3 install ninja cmake setuptools wheel scikit_build_core```<br/>
```git clone https://github.com/pytorch/torchcodec.git && cd torchcodec```<br/>
```export PKG_CONFIG_PATH="$HOME/ffmpeg_local_build/lib/pkgconfig:$PKG_CONFIG_PATH"```<br/>
```export CUDACXX=/usr/local/cuda/bin/nvcc```<br/>
```I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 TORCHCODEC_BUILD_IMAGE=0 pip install . --no-build-isolation -v```
9. Ejecutar el benchmark:<br/>
```python3 src/main.py <acelerador> <modelo> <num_muestras>```

### Notas adicionales
La carpeta `src/data` no se copia dentro de la imagen para reducir su tamaño. Debe existir en el host y montarse al arrancar el contenedor.
La carpeta `results` también debe montarse si se quiere conservar los CSV generados fuera del contenedor.
Los modelos específicos para un acelerador en concreto se encuentran en la carpeta `src/data/models/<acelerador>` y siguen la misma estructura de carpetas que los modelos generales.