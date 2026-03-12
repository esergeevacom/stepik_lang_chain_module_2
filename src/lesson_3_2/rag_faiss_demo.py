import argparse
import logging
from pathlib import Path
from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pydantic import BaseModel, Field, HttpUrl, ValidationError


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class SourceItem(BaseModel):
    source_type: Literal["pdf", "markdown", "html"]
    value: str


class SearchResultItem(BaseModel):
    rank: int
    score: float
    text: str
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class AppConfig(BaseModel):
    sources: list[SourceItem]
    index_dir: Path = Path("./faiss_index")
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 3


def build_demo_config() -> AppConfig:
    """
    Реальные источники по умолчанию:
    - LangChain overview page
    - Requests README из GitHub (raw markdown)

    При желании можно заменить на свои PDF / markdown / html источники.
    """
    return AppConfig(
        sources=[
            SourceItem(
                source_type="html",
                value="https://python.langchain.com/docs/introduction/",
            ),
            SourceItem(
                source_type="markdown",
                value="https://raw.githubusercontent.com/psf/requests/main/README.md",
            ),
        ]
    )


def load_single_source(source: SourceItem) -> list[Document]:
    if source.source_type == "pdf":
        loader = PyMuPDFLoader(source.value)
        documents = loader.load()
        logger.info("Loaded PDF: %s | documents=%s", source.value, len(documents))
        return documents

    if source.source_type == "html":
        _validate_url(source.value)
        loader = WebBaseLoader(web_paths=(source.value,))
        documents = loader.load()
        logger.info("Loaded HTML: %s | documents=%s", source.value, len(documents))
        return documents

    if source.source_type == "markdown":
        if _looks_like_url(source.value):
            _validate_url(source.value)
            loader = WebBaseLoader(web_paths=(source.value,))
        else:
            loader = TextLoader(source.value, encoding="utf-8")
        documents = loader.load()
        logger.info("Loaded Markdown: %s | documents=%s", source.value, len(documents))
        return documents

    raise ValueError(f"Unsupported source type: {source.source_type}")


def load_documents(sources: list[SourceItem]) -> list[Document]:
    documents: list[Document] = []
    for source in sources:
        loaded_documents = load_single_source(source)
        documents.extend(loaded_documents)

    if not documents:
        raise ValueError("No documents were loaded.")

    logger.info("Total loaded documents: %s", len(documents))
    return documents


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into chunks: %s", len(chunks))
    return chunks


def build_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    logger.info("Loading embedding model: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_and_save_index(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    index_dir: Path,
) -> FAISS:
    logger.info("Building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    logger.info("FAISS index saved to: %s", index_dir.resolve())
    return vectorstore


def load_saved_index(index_dir: Path, embeddings: HuggingFaceEmbeddings) -> FAISS:
    if not index_dir.exists():
        raise FileNotFoundError(f"Index directory does not exist: {index_dir}")

    logger.info("Loading FAISS index from: %s", index_dir.resolve())
    return FAISS.load_local(
        folder_path=str(index_dir),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def semantic_search(
    vectorstore: FAISS,
    query: str,
    *,
    top_k: int,
) -> list[SearchResultItem]:
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    output: list[SearchResultItem] = []
    for rank, (document, score) in enumerate(results, start=1):
        output.append(
            SearchResultItem(
                rank=rank,
                score=float(score),
                text=document.page_content,
                metadata=_normalize_metadata(document.metadata),
            )
        )
    return output


def print_results(results: list[SearchResultItem]) -> None:
    if not results:
        print("No results found.")
        return

    for item in results:
        print("=" * 100)
        print(f"Rank: {item.rank}")
        print(f"Score: {item.score:.4f}")
        print(f"Metadata: {item.metadata}")
        print("Text:")
        print(item.text.strip())
        print()


def _looks_like_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _validate_url(value: str) -> None:
    try:
        HttpUrl(value)
    except ValidationError as exc:
        raise ValueError(f"Invalid URL: {value}") from exc


def _normalize_metadata(
    metadata: dict[str, object],
) -> dict[str, str | int | float | bool | None]:
    normalized: dict[str, str | int | float | bool | None] = {}
    for key, value in metadata.items():
        if isinstance(value, str | int | float | bool) or value is None:
            normalized[str(key)] = value
        else:
            normalized[str(key)] = str(value)
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Индексация документов и семантический поиск по эмбеддингам через FAISS.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="How does LangChain help build LLM applications?",
        help="Поисковый запрос для проверки индекса.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("./faiss_index"),
        help="Папка для сохранения FAISS-индекса.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=700,
        help="Размер чанка.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=120,
        help="Перекрытие чанков.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Сколько результатов вернуть.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = build_demo_config().model_copy(
        update={
            "index_dir": args.index_dir,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "top_k": args.top_k,
        }
    )

    documents = load_documents(config.sources)
    chunks = split_documents(
        documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    embeddings = build_embeddings(config.embedding_model_name)

    build_and_save_index(
        documents=chunks,
        embeddings=embeddings,
        index_dir=config.index_dir,
    )

    restored_vectorstore = load_saved_index(
        index_dir=config.index_dir,
        embeddings=embeddings,
    )

    results = semantic_search(
        vectorstore=restored_vectorstore,
        query=args.query,
        top_k=config.top_k,
    )
    print_results(results)


if __name__ == "__main__":
    main()
