"""
Ingestion script for the RAG pipeline.

Reads every PDF in the documents/ folder, splits them into small chunks,
tags each chunk with some metadata (source, page, embedding model, ...),
embeds them, and stores everything in the local Chroma vector database
so rag.py can search over it later.

Run it directly with `python ingest.py` whenever the documents/ folder
changes and the database needs to be rebuilt.
"""

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
import uuid

from embeddings import EMBEDDING_MODEL, load_embedding_model

# Paths: documents/ holds the source PDFs, database/ is where Chroma
# persists the resulting vector store.
PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
DATABASE_DIR = PROJECT_ROOT / "database"

documents = []

# Load every PDF in documents/ - PyMuPDFLoader turns each page into its
# own Document, with the page number already in its metadata.
for pdf_path in DOCUMENTS_DIR.glob("*.pdf"):
    loader = PyMuPDFLoader(str(pdf_path))
    documents.extend(loader.load())


print(f"Total number of documents: {len(documents)}")

#Meta data

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
VERSION = "v1"

# Split each page into smaller chunks so retrieval can return focused,
# relevant snippets instead of whole pages.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
    )

chunks = text_splitter.split_documents(documents)
print(f"Total number of chunks: {len(chunks)}")

# Attach metadata to every chunk (a unique id, where it came from, and
# which embedding model/chunk settings produced it) so it can be traced
# back and filtered on later.
for i, chunk in enumerate(chunks):

    chunk.metadata.update(
        {
            "chunk_id": str(uuid.uuid4()),
            "source": chunk.metadata.get("source"),
            "page": chunk.metadata.get("page"),
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "version": VERSION,
            "created_at": datetime.now().isoformat()
        }
    )




#Save the embeddings to a vector store

# Embed every chunk and add it to the persisted Chroma database
# this is the same database rag.py opens later to answer questions.
embedding_model = load_embedding_model()

# multilingual-e5 embeddings are normalized and meant to be compared
# with cosine similarity
db = Chroma(
    persist_directory=str(DATABASE_DIR),
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

db.add_documents(chunks)