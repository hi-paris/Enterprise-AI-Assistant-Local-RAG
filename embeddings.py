"""
Shared embedding model for ingest.py and rag.py.

multilingual-e5 models are trained with distinct "query: " and
"passage: " prefixes on the text they embed, and are meant to be
compared with cosine similarity (hence normalize_embeddings=True).
Centralizing this here means both scripts always embed the same way.
"""

from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


class E5Embeddings(HuggingFaceEmbeddings):
    """
    Adds the "query: "/"passage: " prefixes multilingual-e5 expects.
    Only the text sent to the model is prefixed - Chroma still stores
    and returns the original, unprefixed chunk content.
    """

    def embed_documents(self, texts):
        return super().embed_documents([f"passage: {text}" for text in texts])

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")


def load_embedding_model():
    return E5Embeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
