from pathlib import Path

HUMAN_ID_COCO = 0 # ID de la categoría "persona" en el dataset COCO
HUMAN_DETECTION_DATA_BBOX_JSON = "human_detection_data_bbox.json" # JSON con las bounding boxes del dataset
HUMAN_DETECTION_DATASET_PATH = "src/data/datasets/human_detection" # Ruta al dataset
EDGETPU_LIB_PATH = "src/data/libs/libedgetpu.so.1" # Ruta a la librería de Edge TPU

def get_model_path(accelerator: str):
    if accelerator == "cpu":
        model_path = "src/data/models/mobilenet/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
    elif accelerator == "gpu":
        model_path = "src/data/models/mobilenet/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
    elif accelerator == "npu":
        model_path = "src/data/models/mobilenet/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
    elif accelerator == "tpu":
        model_path = "src/data/models/tpu/mobilenet/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
    else:
        raise ValueError(f"Tipo de acelerador no válido: {accelerator}")
    return Path(model_path)