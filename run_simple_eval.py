from __future__ import annotations

import json
import re
from typing import List, Dict, Any

import pandas as pd


INPUT_JSON = "eval_outputs_dense_with_answers.json"
OUT_CSV = "eval_results_dense.csv"
OUT_SUMMARY = "eval_summary_dense.json"


STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those",
    "as", "at", "from", "it", "its", "into", "than", "then",
    "how", "what", "when", "which", "who", "whom", "why",
    "do", "does", "did", "can", "could", "should", "would",
    "have", "has", "had", "may", "might", "will", "shall",
    "their", "there", "about", "over", "under", "through", "during"
}


def load_eval_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in STOPWORDS]


def token_set(text: str) -> set:
    return set(tokenize(text))


def f1_overlap(text1: str, text2: str) -> float:
    """
    Simple token-overlap F1 score in [0,1]
    """
    s1 = token_set(text1)
    s2 = token_set(text2)

    if not s1 or not s2:
        return 0.0

    overlap = len(s1 & s2)
    if overlap == 0:
        return 0.0

    precision = overlap / len(s1)
    recall = overlap / len(s2)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def score_faithfulness_proxy(answer: str, ground_truth: str) -> float:
    """
    Proxy for faithfulness:
    compare generated answer to ground truth with token-overlap F1.
    """
    return f1_overlap(answer, ground_truth)


def score_context_precision_proxy(question: str, contexts: List[str]) -> float:
    """
    Proxy for context precision:
    average token-overlap F1 between question and each retrieved chunk.
    """
    if not contexts:
        return 0.0

    scores = [f1_overlap(question, ctx) for ctx in contexts]
    return sum(scores) / len(scores)


def main():
    print("Loading eval outputs...")
    rows = load_eval_data(INPUT_JSON)
    print(f"Loaded {len(rows)} rows")

    results = []
    for i, row in enumerate(rows, start=1):
        question = row["question"]
        answer = row.get("generated_answer", "")
        ground_truth = row["ground_truth"]
        contexts = row.get("retrieved_contexts", [])

        faithfulness = score_faithfulness_proxy(answer, ground_truth)
        context_precision = score_context_precision_proxy(question, contexts)

        new_row = dict(row)
        new_row["faithfulness"] = round(faithfulness, 4)
        new_row["context_precision"] = round(context_precision, 4)
        results.append(new_row)

        print(f"[{i}/{len(rows)}] done")

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)

    summary = {
        "num_questions": len(df),
        "avg_faithfulness": round(float(df["faithfulness"].mean()), 4),
        "avg_context_precision": round(float(df["context_precision"].mean()), 4),
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_SUMMARY}")
    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()
