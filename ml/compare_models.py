"""Compare all shipped models on two benchmarks and save one artifact.

  - held-out test split (data/processed/test.csv): in-distribution quality
    incl. the hard "hateful but not sexist" negatives
  - HateCheck (data/processed/hatecheck_eval.csv): out-of-domain functional
    probe; every model is evaluated at threshold 0.5 here

Output: reports/model_comparison.csv

Run:  python ml/compare_models.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import predict_df  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    "sexism-distilbert-v2",
    "local_5",
    "local_3",
    "collab_5",
    "collab_7",
    "2_collab_3",
]


def score(df: pd.DataFrame, prob) -> dict:
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(df["label"], pred)), 3),
        "precision": round(float(precision_score(df["label"], pred, zero_division=0)), 3),
        "recall": round(float(recall_score(df["label"], pred)), 3),
        "f1": round(float(f1_score(df["label"], pred)), 3),
        "roc_auc": round(float(roc_auc_score(df["label"], prob)), 3),
    }


def main() -> None:
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    hatecheck = pd.read_csv(ROOT / "data" / "processed" / "hatecheck_eval.csv")
    hard_neg = test["source"] == "mhs_hateful_nonsexist"

    rows = []
    for name in MODELS:
        model_dir = ROOT / "models" / name
        if not (model_dir / "config.json").exists():
            print(f"skip {name}: not found")
            continue
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval().cuda() if torch.cuda.is_available() else model.eval()

        t_prob = predict_df(test, tokenizer, model, 128)
        h_prob = predict_df(hatecheck, tokenizer, model, 128)

        row = {"model": name, **{f"test_{k}": v for k, v in score(test, t_prob).items()}}
        row["test_hardneg_accuracy"] = round(float((test.loc[hard_neg, "label"] == (t_prob[hard_neg.to_numpy()] >= 0.5).astype(int)).mean()), 3)
        row.update({f"hatecheck_{k}": v for k, v in score(hatecheck, h_prob).items()})
        rows.append(row)
        print(row)
        if torch.cuda.is_available():
            del model
            torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)
    print(f"saved reports/model_comparison.csv\n{out.to_string(index=False)}")


if __name__ == "__main__":
    main()
