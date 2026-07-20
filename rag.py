"""
The RAG assistant: retrieves relevant chunks from the local Chroma database
and asks a local Ollama model to answer using them as context.

Run directly (`python rag.py`) for an interactive chat loop.
"""

from langchain_chroma import Chroma

import ollama

from embeddings import load_embedding_model


LLM_MODEL = "qwen2.5:3b"

# Cosine distance above this is treated as "no relevant match".
MAX_DISTANCE = 0.18

# How many past conversation turns to keep, to avoid exceeding the model's context window.
MAX_HISTORY = 10


embedding_model = load_embedding_model()

# Same directory ingest.py writes to - this is where the document chunks live.
db = Chroma(
    persist_directory="./database",
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

# System prompt is French-only and answer-from-context-only because the sample
# documents are French corporate docs. 
conversation = [
    {
        "role": "system",
        "content":
        """
        Tu es un assistant d'entreprise.

        Règles :
        - Réponds uniquement en utilisant le contexte fourni.
        - N'invente aucune information.
        - Réponds toujours en français.
        - Si l'information est manquante, dis :
        'Je ne sais pas d'après les documents disponibles.'
        """
    }
]


def retrieve_documents(question):
    """Return the top-k chunks for a question, dropping matches weaker than MAX_DISTANCE."""
    results = db.similarity_search_with_score(question, k=3)
    return [(doc, score) for doc, score in results if score <= MAX_DISTANCE]


def build_context_prompt(question, results):
    """Turn retrieved (doc, score) pairs and the question into the final LLM prompt."""
    context = ""

    for doc, score in results:
        context += f"""

SOURCE :
{doc.metadata.get("source")}

PAGE :
{doc.metadata.get("page")}

CONTENU :
{doc.page_content}

------------------------
"""

    return f"""

Utilise UNIQUEMENT le contexte ci-dessous pour répondre.

Réponds en français.

Si la réponse ne se trouve pas dans le contexte :
"Je ne sais pas d'après les documents disponibles."


Contexte :

{context}


Question :

{question}

"""


def generate_answer(messages):
    """Call the local Ollama model and return its answer text."""
    response = ollama.chat(model=LLM_MODEL, messages=messages)
    return response["message"]["content"]


def ask(question):
    """
    Run the full RAG pipeline for one question: retrieve context, prompt the
    model, and update conversation memory. Returns the answer text (or a
    fallback message if nothing matched closely enough).
    """
    global conversation

    results = retrieve_documents(question)

    if not results:
        return "I don't know based on the available documents."

    rag_prompt = build_context_prompt(question, results)

    # Prepend conversation history so far, without permanently storing the
    # full rag_prompt itself in memory - only the question (without context) and answer get appended below.
    messages = conversation.copy()
    messages.append({"role": "user", "content": rag_prompt})

    answer = generate_answer(messages)

    conversation.append({"role": "user", "content": question})
    conversation.append({"role": "assistant", "content": answer})
    # Keep the system prompt plus the last MAX_HISTORY turns.
    conversation[:] = [conversation[0]] + conversation[-MAX_HISTORY:]

    return answer


if __name__ == "__main__":
    while True:
        question = input("You: ")

        if question.lower() == "exit":
            break

        print("Assistant:", ask(question))
