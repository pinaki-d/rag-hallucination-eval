import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Paths
PAPERS_DIR = Path("data/papers")
CHROMA_DIR = Path("data/chroma_db")

# Embedding model — runs on CPU, no API key needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    """Returns the sentence-transformers embedding model."""
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)


def load_and_chunk_pdfs(chunk_size: int = 512, chunk_overlap: int = 50):
    """
    Loads all PDFs from data/papers/, splits them into chunks.

    Args:
        chunk_size: number of characters per chunk
        chunk_overlap: overlap between consecutive chunks

    Returns:
        List of Document objects with text and metadata
    """
    if not PAPERS_DIR.exists():
        raise FileNotFoundError(f"Papers directory not found: {PAPERS_DIR}")

    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {PAPERS_DIR}")

    print(f"Found {len(pdf_files)} PDF files")

    all_docs = []
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        all_docs.extend(pages)

    print(f"Total pages loaded: {len(all_docs)}")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Total chunks created: {len(chunks)} (chunk_size={chunk_size})")

    return chunks


def build_vectorstore(chunks, collection_name: str = "papers"):
    """
    Embeds chunks and stores in ChromaDB.

    Args:
        chunks: list of Document objects
        collection_name: name for the ChromaDB collection

    Returns:
        Chroma vectorstore instance
    """
    embeddings = get_embeddings()

    # Persist to disk so we don't re-embed every time
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR)
    )
    print(f"Vectorstore built with {len(chunks)} chunks")
    return vectorstore


def load_vectorstore(collection_name: str = "papers"):
    """
    Loads an existing ChromaDB vectorstore from disk.
    Call this after build_vectorstore() has been run once.
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    return vectorstore


def ingest_pipeline(chunk_size: int = 512):
    """
    Full ingestion pipeline: load PDFs → chunk → embed → store.
    Run this once to build the vectorstore.
    """
    print(f"\n=== Starting ingestion (chunk_size={chunk_size}) ===")
    chunks = load_and_chunk_pdfs(chunk_size=chunk_size)
    vectorstore = build_vectorstore(chunks)
    print("=== Ingestion complete ===\n")
    return vectorstore


if __name__ == "__main__":
    # Run this script directly to build the vectorstore
    # python -m rag.ingestor
    ingest_pipeline(chunk_size=512)
