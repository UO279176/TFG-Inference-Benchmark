import argparse
import csv
from pathlib import Path

import numpy as np
from ai_edge_quantizer import quantizer, recipe
from transformers import LlamaTokenizerFast

SEQUENCE_LENGTH = 128
PROMPT_PREFIX = "Summarize the following news article:\n\n"
CALIBRATION_LENGTHS = (16, 32, 48, 64, 78, 96, 128)


def generate_calibration_data(
    dataset_path: Path,
    tokenizer_path: Path,
    num_samples: int = 100,
) -> list[dict[str, np.ndarray]]:
    tokenizer = LlamaTokenizerFast.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples: list[dict[str, np.ndarray]] = []
    with dataset_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for sample_index, row in enumerate(reader):
            article = (row.get("article") or "").strip()
            if not article:
                continue

            calibration_length = CALIBRATION_LENGTHS[sample_index % len(CALIBRATION_LENGTHS)]
            tokenized = tokenizer(
                PROMPT_PREFIX + article,
                return_tensors="np",
                truncation=True,
                max_length=calibration_length,
            )
            input_ids = np.zeros((1, SEQUENCE_LENGTH), dtype=np.int64)
            attention_mask = np.zeros((1, SEQUENCE_LENGTH), dtype=np.int64)
            token_count = tokenized["input_ids"].shape[1]
            input_ids[0, :token_count] = tokenized["input_ids"][0]
            attention_mask[0, :token_count] = tokenized["attention_mask"][0]
            samples.append(
                {
                    "args_0": input_ids,
                    "args_1": attention_mask,
                }
            )
            if len(samples) >= num_samples:
                break

    if not samples:
        raise RuntimeError(f"No se encontraron artículos válidos en {dataset_path}")
    return samples


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Cuantiza TinyLlama con textos reales")
    parser.add_argument("--float-model", type=Path, default=Path("tinyllama.tflite"))
    parser.add_argument("--output-model", type=Path, default=Path("tinyllama_quantized.tflite"))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "src/data/datasets/cnn_dailymail_news/validation.csv",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=project_root / "src/data/models/tinyllama",
    )
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    calibration_samples = generate_calibration_data(
        dataset_path=args.dataset,
        tokenizer_path=args.tokenizer,
        num_samples=args.samples,
    )
    print(f"Usando {len(calibration_samples)} muestras reales para calibración")

    qt = quantizer.Quantizer(str(args.float_model))
    qt.load_quantization_recipe(recipe.static_wi8_ai8())
    calibration_result = qt.calibrate({"serving_default": calibration_samples})
    qt.quantize(calibration_result).export_model(str(args.output_model))
    print(f"Modelo cuantizado guardado en {args.output_model}")


if __name__ == "__main__":
    main()