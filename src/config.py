import argparse

args = None

def init_argparse(accelerators_str, models_str):
    global args
    parser = argparse.ArgumentParser()
    
    parser.add_argument("accelerator", choices=accelerators_str, help="Acelerador a utilizar para la inferencia")
    parser.add_argument("model", choices=models_str, help="Modelo a utilizar para la inferencia")
    parser.add_argument("num_samples", type=int, help="Número de muestras a procesar del dataset")
    parser.add_argument("-v", "--verbose", action="store_true", help="Muestra los resultados de las predicciones por cada muestra")

    args = parser.parse_args()
    return args

def get_args():
    """
    Devuelve los argumentos.
    """
    if args is None:
        raise ValueError("Argumentos no inicializados, llama primero a init_argparse()")
    return args
    
def printv(string: str):
    """
    Imprime un mensaje dado solo si verbose está activado.
    """
    if get_args().verbose:
        print(string)