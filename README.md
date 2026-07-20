# Enterprise AI Assistant — Local RAG

A question-answering assistant that runs **entirely on your own machine** — no cloud API, no external vector database. Point it at a folder of internal documents (HR policies, security procedures, contracts, product docs...) and ask it questions. It finds the relevant passages and uses a local LLM (via [Ollama](https://ollama.com)) to write the answer.

This project is here to answer one question: **how well does RAG work when everything runs locally?** It comes with an evaluation script so you can measure retrieval quality and speed, and compare different local models.

The corpus used for the experiments is entirely made up of synthetic
documentation generated with Claude, simulating the different types of
documents produced by a company, NovaCloud, which offers an industrial IoT
monitoring and predictive maintenance solution aimed at SMEs and mid-sized
manufacturing companies. The corpus in `documents/` contains 10 documents
written in French (33 pages across the 10 documents, for a total of 500 KB),
representative of the different types of content a company in this sector
actually produces (HR, technical, contractual, quality, security).

## How it works

```
documents/*.pdf  →  ingest.py  →  database/ (Chroma vector store)  →  rag.py  →  answer
                        ↑                                                ↑
                        └──────────────── embeddings.py ─────────────────┘
```

- **`embeddings.py`** holds the embedding model both scripts share (`intfloat/multilingual-e5-large`), plus the `query:`/`passage:` prefixes it needs. Keeping this in one place means `ingest.py` and `rag.py` always embed text the same way.
- **`ingest.py`** reads every PDF in `documents/`, breaks the text into chunks, turns them into embeddings, and stores them in the local Chroma database under `database/`.
- **`rag.py`** is the assistant itself. For each question it embeds the question, fetches the most relevant chunks from Chroma, and asks a local Ollama model to answer using that context.
- **`rag_eval.py`** is a small test suite — it runs a fixed list of questions through `rag.py` and reports how accurate and how fast the answers are.
- **`eval_sets.py`** holds the test questions themselves (`EVALUATION_SET_simple`, `EVALUATION_SET_medium`, `EVALUATION_SET_advanced`), kept separate from the evaluation logic so they're easy to extend.

## Project structure

```
.
├── documents/            # PDFs to ingest (sample: French corporate docs)
├── database/             # Chroma vector store, built by ingest.py
├── embeddings.py         # Shared embedding model
├── ingest.py             # Builds the vector database from documents/
├── rag.py                # The assistant itself (chat loop)
├── rag_eval.py           # Checks retrieval quality and speed
├── eval_sets.py          # Test questions used by rag_eval.py
├── pyproject.toml / uv.lock   # Dependencies, managed with uv
└── requirements.txt      # Exported from uv.lock, for tools that don't speak uv
```

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) to manage the environment
- [Ollama](https://ollama.com) installed and running (`ollama serve`, or just open the desktop app)

## Getting started

```bash
# Install dependencies
uv sync

# Pull the default model
ollama pull qwen2.5:3b

# Build the vector database from your documents
uv run python ingest.py

# Start chatting
uv run python rag.py
```

Type a question and you'll get an answer, with its sources printed above it. Type `exit` to quit.

Only re-run `ingest.py` when the files in `documents/` change. It *adds* to the existing database rather than replacing it, so ingesting the same PDFs twice creates duplicates — delete the `database/` folder first if you want a clean rebuild. A smarter approach (skip unchanged files, replace changed ones) would avoid that, but that's out of scope here.

## Configuration

Most settings live in `rag.py`. `EMBEDDING_MODEL` is the exception — it lives in `embeddings.py` since `ingest.py` needs it too:

| Setting | Default | Where | What it does |
|---|---|---|---|
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | `embeddings.py` | Model used to embed both documents and questions. If you change it, re-run `ingest.py` against a clean `database/` — old and new embeddings can't be compared |
| `LLM_MODEL` | `qwen2.5:3b` | `rag.py` | Ollama model used to generate answers |
| `MAX_DISTANCE` | `0.18` | `rag.py` | How close a match needs to be before it's used — matches weaker than this are treated as "no answer found". Tuned for `multilingual-e5-large`: real matches scored ~0.09-0.15, unrelated questions scored ~0.21-0.26 |
| `MAX_HISTORY` | `10` | `rag.py` | How many past conversation turns to remember |

## Trying different models

Any chat-capable Ollama model works — you're not stuck with `qwen2.5:3b`. To compare a few:

```bash
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull qwen2.5:14b
```

Rough memory usage (4-bit quantized, the Ollama default), so you know what your machine can handle:

| Model | Disk size | RAM while loaded |
|---|---|---|
| `qwen2.5:3b` | ~2 GB | ~2–3 GB |
| `qwen2.5:7b` | ~4.5 GB | ~5–6 GB |
| `qwen2.5:14b` | ~9 GB | ~10–12 GB |

Check what's actually on your machine with:
```bash
ollama list   # models on disk
ollama ps     # what's currently loaded, with live RAM/VRAM usage
```

To switch models, change one line in `rag.py`:
```python
LLM_MODEL = "qwen2.5:7b"
```

Then re-run `rag.py` or `rag_eval.py` — nothing else to change. For a fair comparison, keep the documents and test questions the same and only change `LLM_MODEL`.

## Evaluating retrieval quality

```bash
uv run python rag_eval.py
```

This runs a set of test questions (in French, matching the sample documents) and prints a report per question plus a summary, covering:

- **Retrieval quality** — source accuracy, page accuracy, keyword recall, hit-rate@k, and raw distance scores 
- **Speed** — retrieval time, generation time, and total time

The test questions live in `eval_sets.py`. Edit `run_evaluation()` in `rag_eval.py` to switch between `EVALUATION_SET_simple`, `EVALUATION_SET_medium`, and `EVALUATION_SET_advanced`.

To see the LLM's real RAM/CPU usage (not the Python process), open a new terminal and run `ollama ps` alongside a test — Ollama runs models in their own process.
