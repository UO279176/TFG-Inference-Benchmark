from pathlib import Path
import csv
import pyarrow.parquet as pq

from PIL import Image

from inference.contracts import AudioSample, ImageSample, TextSample


class ImageNetFolderDataset:
    def __init__(self, dataset_folder_path: Path):
        self.dataset_folder_path = dataset_folder_path

    def iter_samples(self, limit: int | None = None) -> list[ImageSample]:
        print(f"Cargando muestras del dataset desde: {self.dataset_folder_path}")
        if not self.dataset_folder_path.exists():
            raise FileNotFoundError(f"No existe el dataset en la ruta: {self.dataset_folder_path}")

        image_files = sorted(self.dataset_folder_path.glob("*.JPEG"))
        if limit is not None:
            image_files = image_files[:limit]

        samples: list[ImageSample] = []
        for image_path in image_files:
            image = Image.open(image_path).convert("RGB")
            samples.append(ImageSample(path=image_path, image=image))

        return samples

class OpenImagesFolderDataset:
    def __init__(self, dataset_folder_path: Path):
        self.dataset_folder_path = dataset_folder_path

    def iter_samples(self, limit: int | None = None) -> list[ImageSample]:
        print(f"Cargando muestras del dataset desde: {self.dataset_folder_path}")
        if not self.dataset_folder_path.exists():
            raise FileNotFoundError(f"No existe el dataset en la ruta: {self.dataset_folder_path}")

        image_files = sorted(self.dataset_folder_path.glob("*.jpg"))
        if limit is not None:
            image_files = image_files[:limit]
        
        samples: list[ImageSample] = []
        for image_path in image_files:
            image = Image.open(image_path).convert("RGB")
            samples.append(ImageSample(path=image_path, image=image))
            
        return samples

class CnnDailyMailDataset:
    def __init__(self, dataset_folder_path: Path):
        self.dataset_folder_path = dataset_folder_path

    def iter_samples(self, limit: int | None = None) -> list[TextSample]:
        print(f"Cargando muestras del dataset desde: {self.dataset_folder_path}")
        if not self.dataset_folder_path.exists():
            raise FileNotFoundError(f"No existe el dataset en la ruta: {self.dataset_folder_path}")

        csv_path = self.dataset_folder_path / "validation.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"No existe el archivo de validación en la ruta: {csv_path}")

        samples: list[TextSample] = []
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                article = (row.get("article") or "").strip()
                if not article:
                    continue

                sample_id = (row.get("id") or f"sample_{len(samples):04d}").strip()
                highlights = (row.get("highlights") or "").strip() or None
                samples.append(TextSample(path=Path(sample_id), prompt=article, reference=highlights))
                
                # print(f"Cargada muestra: {sample_id} (artículo de {len(article.split())} palabras)")

                if limit is not None and len(samples) >= limit:
                    break

        return samples


class StableDiffusionPromptsDataset:
    def __init__(self, dataset_folder_path: Path):
        self.dataset_folder_path = dataset_folder_path

    def iter_samples(self, limit: int | None = None) -> list[TextSample]:
        print(f"Cargando muestras del dataset desde: {self.dataset_folder_path}")
        if not self.dataset_folder_path.exists():
            raise FileNotFoundError(f"No existe el dataset en la ruta: {self.dataset_folder_path}")

        prompts_path = self.dataset_folder_path / "prompts.txt"
        if not prompts_path.exists():
            raise FileNotFoundError(f"No existe el archivo de prompts en la ruta: {prompts_path}")

        samples: list[TextSample] = []
        with prompts_path.open("r", encoding="utf-8") as prompts_file:
            for line_number, raw_line in enumerate(prompts_file, start=1):
                prompt = raw_line.strip()
                if not prompt:
                    continue

                sample_id = f"prompt_{line_number:03d}"
                samples.append(TextSample(path=Path(sample_id), prompt=prompt))

                if limit is not None and len(samples) >= limit:
                    break

        return samples


class LibriSpeechParquetDataset:
    def __init__(self, dataset_folder_path: Path):
        self.dataset_folder_path = dataset_folder_path
        self.extracted_audio_path = self.dataset_folder_path / "_extracted_audio"

    def _resolve_transcript(self, row: dict) -> str | None:
        for key in ("text", "sentence", "transcript", "normalized_text"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _resolve_audio_path(self, row: dict, sample_id: str) -> Path:
        audio = row.get("audio")
        if isinstance(audio, dict):
            source_path = audio.get("path")
            if isinstance(source_path, str) and source_path.strip():
                candidate = Path(source_path)
                if candidate.is_absolute() and candidate.exists():
                    return candidate

                dataset_candidate = self.dataset_folder_path / candidate
                if dataset_candidate.exists():
                    return dataset_candidate

            raw_bytes = audio.get("bytes")
            if isinstance(raw_bytes, (bytes, bytearray)) and len(raw_bytes) > 0:
                self.extracted_audio_path.mkdir(parents=True, exist_ok=True)
                extracted_path = self.extracted_audio_path / f"{sample_id}.flac"
                extracted_path.write_bytes(bytes(raw_bytes))
                return extracted_path

        raise ValueError("Fila de LibriSpeech sin audio compatible: se esperaba audio.path o audio.bytes")

    def iter_samples(self, limit: int | None = None) -> list[AudioSample]:
        print(f"Cargando muestras del dataset desde: {self.dataset_folder_path}")
        if not self.dataset_folder_path.exists():
            raise FileNotFoundError(f"No existe el dataset en la ruta: {self.dataset_folder_path}")

        parquet_files = sorted(self.dataset_folder_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No se encontró ningún archivo .parquet en: {self.dataset_folder_path}")

        samples: list[AudioSample] = []
        for parquet_path in parquet_files:
            parquet_file = pq.ParquetFile(parquet_path)
            for batch in parquet_file.iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    sample_id = str(row.get("id") or f"sample_{len(samples):06d}")
                    transcript = self._resolve_transcript(row)

                    try:
                        audio_path = self._resolve_audio_path(row, sample_id)
                    except ValueError:
                        continue

                    samples.append(
                        AudioSample(
                            path=Path(sample_id),
                            audio_path=audio_path,
                            reference=transcript,
                        )
                    )

                    if limit is not None and len(samples) >= limit:
                        return samples

        return samples