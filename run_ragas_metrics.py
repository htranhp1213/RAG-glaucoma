from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference
from ragas.llms import LangchainLLMWrapper

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline


@dataclass
class MetricsConfig:
    input_json: str = "eval_outputs_dense_with_answers.json"
    out_csv: str = "eval_results_dense.csv"
    out_summary_json: str = "eval_summary_dense.json"

    # evaluator model
    eval_llm_model: str = "google/flan-t5-base"
    max_new_tokens: int = 128


def load_eval_outputs(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_evaluator_llm(cfg: MetricsConfig):
    """
    Build a local evaluator LLM wrapper for RAGAS.
    """
    hf_pipe = pipeline(
        task="text2text-generation",
        model=cfg.eval_llm_model,
        tokenizer=cfg.eval_llm_model,
        max_new_tokens=cfg.max_new_tokens,
    )
    lc_llm = HuggingFacePipeline(pipeline=hf_pipe)
    return LangchainLLMWrapper(lc_llm)


def convert_to_ragas_samples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    RAGAS-compatible samples.
    """
    samples = []

    for row in rows:
        samples.append(
            {
                "user_input": row["question"],
                "response": row["generated_answer"],
                "retrieved_contexts": row["retrieved_contexts"],
                "reference": row["ground_truth"],
            }
        )

    return samples


def score_with_ragas(rows: List[Dict[str, Any]], cfg: MetricsConfig) -> pd.DataFrame:
    evaluator_llm = build_evaluator_llm(cfg)

    ragas_samples = convert_to_ragas_samples(rows)
    dataset = EvaluationDataset.from_list(ragas_samples)

    metrics = [
        Faithfulness(llm=evaluator_llm),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
    ]

    result = evaluate(dataset=dataset, metrics=metrics)
    result_df = result.to_pandas()

    # merge original metadata columns back in
    original_df = pd.DataFrame(rows).reset_index(drop=True)
    result_df = result_df.reset_index(drop=True)

    merged = pd.concat([original_df, result_df], axis=1)
    return merged


def build_summary(scored_df: pd.DataFrame) -> Dict[str, Any]:
    faith_col = "faithfulness"
    cp_col = "llm_context_precision_with_reference"

    summary = {
        "retriever_type": scored_df["retriever_type"].iloc[0] if "retriever_type" in scored_df.columns and len(scored_df) > 0 else "unknown",
        "num_questions": int(len(scored_df)),
        "top_k": int(scored_df["top_k"].iloc[0]) if "top_k" in scored_df.columns and len(scored_df) > 0 else None,
        "avg_faithfulness": float(scored_df[faith_col].mean()) if faith_col in scored_df.columns else None,
        "avg_context_precision": float(scored_df[cp_col].mean()) if cp_col in scored_df.columns else None,
    }
    return summary


def save_outputs(scored_df: pd.DataFrame, summary: Dict[str, Any], cfg: MetricsConfig):
    scored_df.to_csv(cfg.out_csv, index=False)
    print(f"Saved: {cfg.out_csv}")

    with open(cfg.out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved: {cfg.out_summary_json}")


def main():
    cfg = MetricsConfig()
    rows = load_eval_outputs(cfg.input_json)
    scored_df = score_with_ragas(rows, cfg)
    summary = build_summary(scored_df)
    save_outputs(scored_df, summary, cfg)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    print("\nPreview of scored rows:")
    preview_cols = [
        c for c in [
            "id",
            "question",
            "faithfulness",
            "llm_context_precision_with_reference",
        ] if c in scored_df.columns
    ]
    if preview_cols:
        print(scored_df[preview_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()