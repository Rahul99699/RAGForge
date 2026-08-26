import ingestion.pdf_loader as pdf_loader

print("LOADED FILE:")
print(pdf_loader.__file__)

print("HAS ingest_pdf:")
print(hasattr(pdf_loader, "ingest_pdf"))

print("FUNCTIONS:")
print([
    name
    for name in dir(pdf_loader)
    if not name.startswith("_")
])
from ingestion.pdf_loader import ingest_pdf
from retrieval.vector_store import VectorStore


def ingest_document(pdf_path: str) -> None:
    print(f"Loading: {pdf_path}")

    chunks = ingest_pdf(pdf_path)

    print(f"Created {len(chunks)} chunks")

    if not chunks:
        print("No text chunks were created.")
        return

    store = VectorStore()

    print("Generating embeddings and storing chunks...")

    store.add_chunks(chunks)

    print("Ingestion completed successfully.")


if __name__ == "__main__":
    ingest_document(
        "sample.pdf"
    )