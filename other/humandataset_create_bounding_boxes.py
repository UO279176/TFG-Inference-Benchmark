import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

IMAGES_DIR = Path("src/data/datasets/human_detection/1")
OUTPUT_JSON = Path("annotations_coco.json")
PRIORITY_TXT = Path("check_priority.txt")
CONF_THRESHOLD = 0.35

model = YOLO("yolo26x.pt")

# Estructura base estándar MS COCO 1.0
coco_output = {
    "info": {"description": "Pre-etiquetado de humanos para benchmark"},
    "licenses": [],
    "categories": [
        {"id": 1, "name": "person", "supercategory": "person"}
    ],
    "images": [],
    "annotations": []
}

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
image_paths = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in valid_extensions])

annotation_id = 1
zero_detections = []

for img_id, img_path in enumerate(tqdm(image_paths, desc="Detectando humanos"), start=1):
    with Image.open(img_path) as img:
        width, height = img.size

    coco_output["images"].append({
        "id": img_id,
        "file_name": img_path.name,
        "width": width,
        "height": height
    })

    # classes=[0] filtra exclusivamente la clase 'person' en modelos Ultralytics COCO
    results = model.predict(
        source=str(img_path),
        classes=[0],
        conf=CONF_THRESHOLD,
        verbose=False
    )[0]

    if len(results.boxes) == 0:
        zero_detections.append(img_path.name)

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w = x2 - x1
        h = y2 - y1

        coco_output["annotations"].append({
            "id": annotation_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
            "area": round(w * h, 2),
            "iscrowd": 0
        })
        annotation_id += 1

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(coco_output, f, indent=2)

with open(PRIORITY_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(zero_detections))

print(f"\nProceso terminado: {len(image_paths)} imágenes procesadas, {annotation_id - 1} cajas generadas.")
print(f"Imágenes con 0 detecciones: {len(zero_detections)} -> guardadas en {PRIORITY_TXT}")