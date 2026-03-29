"""
Paquete de inferencia: Define la interfaz común para ejecutar
inferencias con diferentes modelos y sus implementaciones específicas.

Contiene:
- `contracts.py`: Define los contratos (interfaces) para los pipelines de modelos y adaptadores de datasets.
- `registry.py`: Implementa un sistema de registro para asociar modelos y datasets con sus respectivas implementaciones de pipelines y adaptadores.
- `pipelines.py`: Implementaciones de pipelines de inferencia para modelos específicos.
- `datasets.py`: Implementaciones de adaptadores para datasets específicos.
- `runner.py`: Define la clase `InferenceRunner` que ejecuta el proceso de inferencia utilizando un pipeline de modelo y un adaptador de dataset.
"""
