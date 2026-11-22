"""
Evaluation script for the RAG pipeline.

For a list of test questions, this checks three things at once:
1. Retrieval quality - did we fetch the right chunk, from the right
   document and page, containing the words we expect?
2. Latency - how long did the full pipeline (retrieval + generation)
   take to answer, using the real ask() function from rag.py?
3. Answer - do the words we expect appear in the generated answer?

Run it directly with `python rag_eval.py`.
"""

import warnings
warnings.filterwarnings("ignore")

import time

import rag
from rag import db, ask, MAX_DISTANCE
from eval_sets import EVALUATION_SET_simple, EVALUATION_SET_medium, EVALUATION_SET_advanced

# Number of chunks retrieved per question. This is the "k" in "top-k retrieval" - the model sees all k chunks when generating an answer, not just the first one.
K = 3

# =====================================
# Evaluation logic
# =====================================

def evaluate_sample(sample):
    """
    Run one test question through the pipeline and score it.

    Checks the top retrieved chunk against what we expected (right
    document, right page, right keywords), checks whether the correct
    document shows up anywhere in the top-k results, then calls the real
    ask() from rag.py to time the full retrieval + generation pipeline
    and checks the expected keywords against its answer.

    Returns a dict with all these scores plus the generated answer,
    ready to be printed or averaged across the whole evaluation set.
    """

    # Ask Chroma directly (not through ask()) so we can inspect the raw
    # top-k matches before any MAX_DISTANCE filtering is applied.
    results = db.similarity_search_with_score(sample["question"], k=K)  # K is the number of chunks retrieved per question.

    top_doc, top_score = results[0]

    top_source = top_doc.metadata.get("source", "")
    top_page = top_doc.metadata.get("page")

    # Did the single best match come from the document/page we expected?
    source_match = sample["expected_source"] in top_source
    page_match = top_page == sample["expected_page"]

    # Do the words we expect to see actually appear in that top chunk?
    top_text = top_doc.page_content.lower()
    found_keywords = [
        kw for kw in sample["expected_keywords"]
        if kw.lower() in top_text
    ]
    keyword_recall = len(found_keywords) / len(sample["expected_keywords"])

    # Even if the top-1 match isn't the expected document, it still counts
    # as a "hit" if the expected document shows up anywhere in the top-k -
    # generate_answer() sees all k chunks, not just the first one.
    hit_in_topk = any(
        sample["expected_source"] in doc.metadata.get("source", "")
        for doc, _ in results
    )

    # ask() drops chunks above MAX_DISTANCE before prompting, so this is the
    # same check restricted to the chunks the model actually receives.
    hit_in_topk_within_distance = any(
        sample["expected_source"] in doc.metadata.get("source", "")
        for doc, score in results
        if score <= MAX_DISTANCE
    )

    # Start every question from the system prompt only, so an earlier question
    # (and its answer) cannot change the next one.
    del rag.conversation[1:]

    # Simple wall-clock latency for the full pipeline (retrieval + generation),
    # reusing rag.py's actual ask() so it reflects real production timing.
    start = time.perf_counter()
    answer = ask(sample["question"])
    latency_ms = (time.perf_counter() - start) * 1000

    # Do the words we expect appear in the generated answer itself?
    answer_text = answer.lower()
    answer_found_keywords = [
        kw for kw in sample["expected_keywords"]
        if kw.lower() in answer_text
    ]
    answer_keyword_recall = len(answer_found_keywords) / len(sample["expected_keywords"])

    return {
        "question": sample["question"],
        "source_match": source_match,
        "page_match": page_match,
        "keyword_recall": keyword_recall,
        "expected_keywords": sample["expected_keywords"],
        "found_keywords": found_keywords,
        "top_score": top_score,
        "all_scores": [score for _, score in results],
        "hit_in_topk": hit_in_topk,
        "hit_in_topk_within_distance": hit_in_topk_within_distance,
        "under_max_distance": top_score <= MAX_DISTANCE,
        "answer_keyword_recall": answer_keyword_recall,
        "answer_found_keywords": answer_found_keywords,
        "answer": answer,
        "latency_ms": latency_ms,
    }


def run_evaluation():
    """
    Score every question in Evaluation set, print a per-question
    breakdown, then print an overall summary (accuracy, recall, hit rate,
    answer keyword recall, average distance, average latency) across the whole set.
    """
    
    #Should be EVALUATION_SET_simple, EVALUATION_SET_medium, or EVALUATION_SET_advanced 
    # depending on which set you want to evaluate.
    report = [evaluate_sample(sample) for sample in EVALUATION_SET_simple]

    total = len(report)

    # Average each metric across all questions to get the headline numbers.
    source_accuracy = sum(r["source_match"] for r in report) / total
    page_accuracy = sum(r["page_match"] for r in report) / total
    keyword_recall = sum(r["keyword_recall"] for r in report) / total
    hit_rate_at_k = sum(r["hit_in_topk"] for r in report) / total
    hit_rate_within = sum(r["hit_in_topk_within_distance"] for r in report) / total
    answer_recall = sum(r["answer_keyword_recall"] for r in report) / total
    avg_top_score = sum(r["top_score"] for r in report) / total
    avg_latency_ms = sum(r["latency_ms"] for r in report) / total

    print("=" * 50)
    print("RAG RETRIEVAL REPORT")
    print("=" * 50)

    for r in report:

        print(f"\nQ: {r['question']}")
        print(f"  Top-1 source match : {r['source_match']}")
        print(f"  Top-1 page match   : {r['page_match']}")
        print(f"  Keyword recall     : {r['keyword_recall']:.0%} ({r['found_keywords']})")
        print(f"  Top-1 distance     : {r['top_score']:.4f} (MAX_DISTANCE={MAX_DISTANCE}, under={r['under_max_distance']})")
        print(f"  All top-{K} scores  : {[f'{s:.4f}' for s in r['all_scores']]}")
        print(f"  Hit in top-{K}      : {r['hit_in_topk']}")
        print(f"  Answer keyword recall: {r['answer_keyword_recall']:.0%} ({r['answer_found_keywords']})")
        print(f"  Latency (end-to-end): {r['latency_ms']:.1f} ms")
        print(f"  Ground truth        : {r['expected_keywords']}")
        print(f"  Generated answer    : {r['answer']}")

    print("\n" + "-" * 50)
    print("SUMMARY")
    print("-" * 50)
    print(f"Source Accuracy (top-1)   : {source_accuracy:.2%}")
    print(f"Page Accuracy (top-1)     : {page_accuracy:.2%}")
    print(f"Keyword Recall (top-1)    : {keyword_recall:.2%}")
    print(f"Hit Rate @ {K} (Recall@{K})  : {hit_rate_at_k:.2%}")
    print(f"Hit Rate @ {K} within MAX_DISTANCE: {hit_rate_within:.2%}")
    print(f"Answer Keyword Recall     : {answer_recall:.2%}")
    print(f"Avg Top-1 Distance        : {avg_top_score:.4f}")
    print(f"Avg Latency (end-to-end)  : {avg_latency_ms:.1f} ms")


if __name__ == "__main__":
    run_evaluation()
