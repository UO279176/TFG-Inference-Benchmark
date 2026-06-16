import nemo.collections.asr as nemo_asr
from pathlib import Path

# 1. Configurar rutas de origen y destino
nemo_model_path = "./src/data/models/rnnt/parakeet-rnnt-1.1b.nemo"
output_dir = Path("./my-onnx-dir")
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Cargar el modelo RNNT
print("Cargando modelo...")
model = nemo_asr.models.ASRModel.restore_from(nemo_model_path)

# 3. Exportar a formato ONNX
print("Exportando a ONNX...")
base_onnx_path = str(output_dir / "rnnt_model.onnx")
model.export(base_onnx_path)

# 4. Extraer el vocabulario
print("Guardando el vocabulario de tokens...")
vocab_path = output_dir / "tokens.txt"
with open(vocab_path, "w", encoding="utf-8") as f:
    # Se añade el token <blk> (blank) al final
    for token in model.tokenizer.vocab:
        f.write(f"{token}\n")

print(f"Conversión completada con éxito en: {output_dir}")