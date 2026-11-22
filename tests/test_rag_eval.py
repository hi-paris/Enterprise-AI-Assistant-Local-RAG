"""Checks for rag_eval.py and the conversation history in rag.py.

No model, database or network: embeddings, Chroma and Ollama are replaced by
small stand-ins, everything else is the real code. Standard library only.
Run: python tests/test_rag_eval.py
"""
import contextlib
import io
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STATE = {"distance": 0.10, "answer": "", "calls": []}

_emb = types.ModuleType("embeddings")
_emb.EMBEDDING_MODEL = "stub"
_emb.load_embedding_model = lambda: object()
sys.modules["embeddings"] = _emb

import eval_sets  # noqa: E402

QUESTIONS = eval_sets.EVALUATION_SET_simple


class _Doc:
    def __init__(self, text, source, page):
        self.page_content, self.metadata = text, {"source": source, "page": page}


class _Chroma:
    def __init__(self, *args, **kwargs):
        pass

    def similarity_search_with_score(self, question, k=3):
        sample = next(s for s in QUESTIONS if s["question"] == question)
        right = _Doc(" ".join(sample["expected_keywords"]),
                     "documents/" + sample["expected_source"], sample["expected_page"])
        other = _Doc("autre sujet", "documents/99_other.pdf", 0)
        return [(right, STATE["distance"]), (other, 0.30), (other, 0.35)][:k]


_lc = types.ModuleType("langchain_chroma")
_lc.Chroma = _Chroma
sys.modules["langchain_chroma"] = _lc


def _chat(model, messages):
    STATE["calls"].append([m["role"] for m in messages])
    answer = STATE["answer"]
    return {"message": {"content": answer(messages) if callable(answer) else answer}}


def _correct_answer(messages):
    """A model that answers each question with its expected keywords."""
    sample = next(s for s in QUESTIONS if s["question"] in messages[-1]["content"])
    return "D'après les documents : " + " et ".join(sample["expected_keywords"]) + "."


_ollama = types.ModuleType("ollama")
_ollama.chat = _chat
sys.modules["ollama"] = _ollama


def run_eval(distance, answer):
    for name in ("rag", "rag_eval"):
        sys.modules.pop(name, None)
    STATE.update(distance=distance, answer=answer, calls=[])
    import rag_eval
    with contextlib.redirect_stdout(io.StringIO()) as out:
        rag_eval.run_evaluation()
    return out.getvalue()


def summary_value(output, label):
    line = next(line for line in output.splitlines() if line.startswith(label))
    return line.split(":", 1)[1].strip()


class RagEvalTests(unittest.TestCase):
    def test_answer_is_scored(self):
        right = run_eval(0.10, _correct_answer)
        wrong = run_eval(0.10, "Je ne sais pas d'après les documents disponibles.")
        self.assertEqual(summary_value(right, "Answer Keyword Recall"), "100.00%")
        self.assertEqual(summary_value(wrong, "Answer Keyword Recall"), "0.00%")

    def test_chunk_beyond_max_distance_is_not_a_hit_within_distance(self):
        out = run_eval(0.20, "peu importe")
        self.assertEqual(summary_value(out, "Hit Rate @ 3 within MAX_DISTANCE"), "0.00%")
        self.assertEqual(summary_value(out, "Hit Rate @ 3 (Recall@3)"), "100.00%")  # raw line unchanged

    def test_chunk_within_max_distance_is_a_hit(self):
        out = run_eval(0.10, "peu importe")
        self.assertEqual(summary_value(out, "Hit Rate @ 3 within MAX_DISTANCE"), "100.00%")

    def test_questions_do_not_share_history(self):
        run_eval(0.10, "ok")
        self.assertEqual([len(roles) for roles in STATE["calls"]], [2, 2, 2])

    def test_system_prompt_is_not_repeated(self):
        for name in ("rag", "rag_eval"):
            sys.modules.pop(name, None)
        STATE.update(distance=0.10, answer="ok", calls=[])
        import rag
        rag.ask(QUESTIONS[0]["question"])
        rag.ask(QUESTIONS[1]["question"])
        self.assertEqual(STATE["calls"][1], ["system", "user", "assistant", "user"])
        self.assertEqual(rag.conversation[0]["role"], "system")
        self.assertEqual([m["role"] for m in rag.conversation[1:]],
                         ["user", "assistant", "user", "assistant"])


if __name__ == "__main__":
    unittest.main()
