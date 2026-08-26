

import hashlib
import re
from pathlib import Path

import pymupdf

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


# ---------------------------------------------------------------------------
# Document utilities
# ---------------------------------------------------------------------------

def create_document_id(pdf_path: str | Path) -> str:
    """Create a stable document ID from the resolved PDF path."""
    path = Path(pdf_path).resolve()

    return hashlib.sha256(
        str(path).encode("utf-8")
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean common artifacts from extracted PDF text."""
    text = text.replace("\x00", " ")

    # Normalize spaces and tabs while preserving newlines.
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# PDF loading
# ---------------------------------------------------------------------------

def load_pdf(pdf_path: str | Path) -> list[dict]:
    """Extract cleaned text from a PDF page by page."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not pdf_path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    document_id = create_document_id(pdf_path)

    pages: list[dict] = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = clean_text(page.get_text("text"))

            if not text:
                continue

            pages.append(
                {
                    "document_id": document_id,
                    "page": page_number,
                    "source": pdf_path.name,
                    "text": text,
                }
            )

    return pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def create_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Create the recursive character text splitter."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[     
            "\n\n",     # Paragraph
            "\n",       # Line
            ". ",       # Sentence
            "? ",
            "! ",
            " ",        # Word
            "",         # Character fallback
        ],
        strip_whitespace=True,
    )


def chunk_text(
    text: str,
    splitter: RecursiveCharacterTextSplitter,
) -> list[str]:
    """Split text into recursive, overlapping chunks."""
    if not text.strip():
        return []

    return splitter.split_text(text)


def ingest_pdf(
    pdf_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Extract a PDF and return chunks with metadata."""

    pages = load_pdf(pdf_path)

    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[dict] = []

    for page in pages:
        page_chunks = chunk_text(
            page["text"],
            splitter,
        )

        for chunk_index, text in enumerate(page_chunks):
            chunks.append(
                {
                    "document_id": page["document_id"],
                    "chunk_id": (
                        f"{page['document_id']}"
                        f"-p{page['page']}"
                        f"-c{chunk_index}"
                    ),
                    "chunk_index": chunk_index,
                    "page": page["page"],
                    "source": page["source"],
                    "text": text,
                }
            )

    return chunks