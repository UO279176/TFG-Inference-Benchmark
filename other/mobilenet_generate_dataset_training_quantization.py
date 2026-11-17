from pathlib import Path
from PIL import Image

coco_dir = Path("./val2017")
selected_paths = []

for img_path in sorted(coco_dir.glob("*.jpg")):
    with Image.open(img_path) as img:
        if img.mode == "RGB":
            selected_paths.append(str(img_path.resolve()))
    if len(selected_paths) == 150:
        break

Path("dataset.txt").write_text("\n".join(selected_paths) + "\n", encoding="utf-8")
print(f"Generado dataset.txt con {len(selected_paths)} imágenes")