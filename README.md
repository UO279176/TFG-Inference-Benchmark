# TFG-Inference-Benchmark
Repositorio para mi TFG: Comparativa de dispositivos SBC de bajo coste para la ejecución de modelos de inferencia de IA.

En esto repositorio se encuentra el código fuente del proyecto y los distintos modelos y datasets a utilizar. El objetivo del proyecto es evaluar el rendimiento de diferentes dispositivos SBC (Single Board Computers) de bajo coste en la ejecución de modelos de inferencia de IA.

## Preparación del entorno y ejecución
Para preparar el entorno de ejecución, se deben seguir los siguientes pasos:
0. Instalar Python 3.12 (recomendada la versión 3.12.10). No usar versiones superiores, no son compatibles.
1. Clonar el repositorio:
```git clone https://github.com/UO279176/TFG-Inference-Benchmark.git```
2. Acceder al directorio del proyecto:
```cd TFG-Inference-Benchmark```
3. Crear un entorno virtual (opcional pero recomendado):
```python -m venv venv```
4. Activar el entorno virtual:
- En Windows:
```venv\Scripts\activate```
- En Linux:
```source venv/bin/activate```
5. Instalar las dependencias necesarias:
```pip install -r requirements.txt```
6. Ejecutar el programa principal:
```python src/main.py <acelerador> <modelo>```.
Ejecutar el programa sin parámetros mostrará la ayuda con las opciones disponibles.