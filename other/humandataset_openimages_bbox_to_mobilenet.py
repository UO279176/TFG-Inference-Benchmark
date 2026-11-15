import csv
import json
from pathlib import Path

INPUT_CSV = Path("other/humandataset-annotations-bbox-openimages-v6.csv")
IMAGES_DIR = Path("src/data/datasets/human_detection/1")
OUTPUT_JSON = Path("human_detection_data_bbox.json")

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
dataset = {}
for img_path in sorted(IMAGES_DIR.iterdir()):
    if img_path.suffix.lower() in valid_extensions:
        dataset[img_path.stem] = {
            "file_name": img_path.name,
            "bbox": []
        }

# Leer el CSV formato OpenImages V6 y reordenar a [ymin, xmin, ymax, xmax]
boxes_loaded = 0
with open(INPUT_CSV, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        image_id = row["ImageID"]
        if image_id not in dataset:
            continue

        ymin = float(row["YMin"])
        xmin = float(row["XMin"])
        ymax = float(row["YMax"])
        xmax = float(row["XMax"])

        dataset[image_id]["bbox"].append([ymin, xmin, ymax, xmax])
        boxes_loaded += 1

# Guardar lista final ordenada por nombre de archivo
final_ground_truth = list(dataset.values())
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_ground_truth, f, indent=2)

print(f"Imágenes indexadas: {len(final_ground_truth)} | Cajas totales cargadas: {boxes_loaded}")
print(f"Archivo bbox guardado en: {OUTPUT_JSON}")