import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

from datasets import load_dataset
from langchain_core.documents import Document
from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    dataset_name: str
    dataset_config: str | None = None
    split: str = "train"
    limit: int | None = None
    min_text_length: int = 100
    remove_duplicates: bool = True
    output_path: str = "prepared.jsonl"


class ProcessingStats(BaseModel):
    rows_loaded: int = 0
    rows_processed: int = 0
    rows_skipped_empty: int = 0
    rows_skipped_short: int = 0
    rows_skipped_duplicates: int = 0
    documents_saved: int = 0


class DocumentPayload(BaseModel):
    page_content: str
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class RAGPreprocessor:
    TEXT_FIELDS: tuple[str, ...] = ("text", "context", "article", "abstract", "page_content")
    META_FIELDS: tuple[str, ...] = ("id", "title", "url", "source", "page", "chunk_id", "chunk_index")

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.stats = ProcessingStats()
        self.logger = self._build_logger()

    def _build_logger(self) -> logging.Logger:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
        return logging.getLogger(self.__class__.__name__)

    def load_rows(self):
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            split=self.config.split,
        )

        for index, row in enumerate(dataset):
            if self.config.limit is not None and index >= self.config.limit:
                break
            self.stats.rows_loaded += 1
            yield row

    def extract_text(self, row: dict) -> str:
        for field_name in self.TEXT_FIELDS:
            field_value = row.get(field_name)
            if isinstance(field_value, str) and field_value.strip():
                return field_value
        return ""

    def clean_text(self, text: str) -> str:
        cleaned_text = text.replace("\u00a0", " ")
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        return cleaned_text.strip()

    def build_metadata(self, row: dict, cleaned_text: str) -> dict[str, str | int | float | bool | None]:
        metadata: dict[str, str | int | float | bool | None] = {
            "dataset_name": self._resolve_dataset_name(),
            "dataset_split": self.config.split,
            "text_length": len(cleaned_text),
            "content_hash": hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest(),
        }

        for field_name in self.META_FIELDS:
            field_value = row.get(field_name)
            if isinstance(field_value, str | int | float | bool):
                metadata[field_name] = field_value

        return metadata

    def build_documents(self) -> list[Document]:
        documents: list[Document] = []
        seen_hashes: set[str] = set()

        for row in self.load_rows():
            raw_text = self.extract_text(row)
            if not raw_text:
                self.stats.rows_skipped_empty += 1
                continue

            cleaned_text = self.clean_text(raw_text)
            if not cleaned_text:
                self.stats.rows_skipped_empty += 1
                continue

            if len(cleaned_text) < self.config.min_text_length:
                self.stats.rows_skipped_short += 1
                continue

            metadata = self.build_metadata(row, cleaned_text)
            content_hash = str(metadata["content_hash"])

            if self.config.remove_duplicates and content_hash in seen_hashes:
                self.stats.rows_skipped_duplicates += 1
                continue

            seen_hashes.add(content_hash)
            documents.append(Document(page_content=cleaned_text, metadata=metadata))
            self.stats.rows_processed += 1

        return documents

    def save(self, documents: list[Document]) -> Path:
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            for document in documents:
                payload = DocumentPayload(
                    page_content=document.page_content,
                    metadata=document.metadata,
                )
                file.write(json.dumps(payload.model_dump(), ensure_ascii=False) + "\n")

        self.stats.documents_saved = len(documents)
        self.logger.info("Процесс завершен: %s", self.stats.model_dump())
        self.logger.info("Сохранено %s документов в %s", len(documents), output_path)
        return output_path

    def process(self) -> list[Document]:
        return self.build_documents()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Пайплайн препроцессинга данных для RAG на основе датасетов Hugging Face"
    )

    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Название датасета для загрузки через load_dataset (например: wikimedia/wikipedia)"
    )

    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Конфигурация датасета (например: 20231101.ru для русской Википедии)"
    )

    parser.add_argument(
        "--split",
        default="train",
        help="Сплит датасета (например: train, validation или train[:100])"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Максимальное количество строк датасета для обработки"
    )

    parser.add_argument(
        "--min-text-length",
        type=int,
        default=100,
        help="Минимальная длина текста чанка. Более короткие тексты будут отфильтрованы"
    )

    parser.add_argument(
        "--remove-duplicates",
        action="store_true",
        help="Удалять дубликаты текстовых чанков"
    )

    parser.add_argument(
        "--output",
        default="prepared.jsonl",
        help="Файл для сохранения подготовленных данных (формат JSONL)"
    )

    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    config = PipelineConfig(
        dataset_name=arguments.dataset_name,
        dataset_config=arguments.dataset_config,
        split=arguments.split,
        limit=arguments.limit,
        min_text_length=arguments.min_text_length,
        remove_duplicates=arguments.remove_duplicates,
        output_path=arguments.output,
    )

    preprocessor = RAGPreprocessor(config)
    documents = preprocessor.process()
    preprocessor.save(documents)


if __name__ == "__main__":
    main()
