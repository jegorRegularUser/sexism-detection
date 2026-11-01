"""Final evaluation: held-out test split + HateCheck functional breakdown.

Produces:
  reports/evaluation.json          - test metrics, per-source accuracy
  reports/confusion_matrix.png     - test-split confusion matrix
  reports/hatecheck_breakdown.csv  - F1/accuracy per HateCheck functionality
  reports/hatecheck_breakdown.png  - bar chart of the same

Run:  python ml/evaluate.py [--model models/sexism-distilbert-v2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


@torch.no_grad()
def predict_df(df: pd.DataFrame, tokenizer, model, max_len: int, batch_size: int = 128):
    device = next(model.parameters()).device
    probs = []
    model.eval()
    for start in range(0, len(df), batch_size):
        batch = df["text"].iloc[start : start + batch_size].tolist()
        enc = tokenizer(batch, truncation=True, max_length=max_len, padding=True,
                        return_tensors="pt").to(device)
        logits = model(**enc).logits
        probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def evaluate_frame(df: pd.DataFrame, prob: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    auc = roc_auc_score(df["label"], prob) if df["label"].nunique() > 1 else None
    report = classification_report(df["label"], pred, output_dict=True, zero_division=0)
    pos = report.get("1") or report.get(1)
    return {
        "n": int(len(df)),
        "accuracy": float(report["accuracy"]),
        "precision": float(pos["precision"]) if pos else None,
        "recall": float(pos["recall"]) if pos else None,
        "f1": float(pos["f1-score"]) if pos else None,
        "roc_auc": float(auc) if auc is not None else None,
        "confusion": confusion_matrix(df["label"], pred).tolist(),
    }


def load_threshold(model_dir: str, override: float | None) -> float:
    """Use the threshold shipped with the model unless explicitly overridden."""
    if override is not None:
        return override
    meta = Path(model_dir) / "threshold.json"
    if meta.exists():
        return float(json.loads(meta.read_text(encoding="utf-8"))["threshold"])
    return 0.5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models" / "sexism-distilbert-v2"))
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--threshold", type=float, default=None,
                   help="override the threshold stored in the model dir")
    args = p.parse_args()
    args.threshold = load_threshold(args.model, args.threshold)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"evaluating {args.model} on {device}")

    results: dict = {"model": args.model, "threshold": args.threshold}

    # ---- held-out test split -------------------------------------------------
    test_df = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    prob = predict_df(test_df, tokenizer, model, args.max_len)
    test_metrics = evaluate_frame(test_df, prob, args.threshold)
    results["test"] = test_metrics
    print("test:", {k: v for k, v in test_metrics.items() if k != "confusion"})

    cm = np.array(test_metrics["confusion"])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title("Test split confusion matrix")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # per-source performance (do we lose accuracy on any data domain?)
    per_source = {}
    for source, part in test_df.groupby("source"):
        idx = part.index
        m = evaluate_frame(test_df.loc[idx], prob[idx], args.threshold)
        per_source[str(source)] = {k: m[k] for k in ("n", "accuracy", "f1")}
    results["test_per_source"] = per_source

    # ---- HateCheck functional breakdown -------------------------------------
    hc = pd.read_csv(ROOT / "data" / "processed" / "hatecheck_eval.csv")
    hc_prob = predict_df(hc, tokenizer, model, args.max_len)
    hc_pred = (hc_prob >= args.threshold).astype(int)
    hc_overall = evaluate_frame(hc, hc_prob, args.threshold)
    results["hatecheck_overall"] = hc_overall
    print("hatecheck:", {k: v for k, v in hc_overall.items() if k != "confusion"})

    rows = []
    for func, part in hc.groupby("functionality"):
        idx = part.index
        y = part["label"]
        pred = hc_pred[idx]
        rows.append({
            "functionality": func,
            "n": len(part),
            "positives": int(y.sum()),
            "accuracy": float((y == pred).mean()),
            "f1": float(f1_score(y, pred, zero_division=0)) if y.nunique() > 1 else None,
            "false_positive_rate": float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)),
            "false_negative_rate": float(((pred == 0) & (y == 1)).sum() / max((y == 1).sum(), 1)),
        })
    breakdown = pd.DataFrame(rows).sort_values("f1")
    breakdown.to_csv(REPORTS / "hatecheck_breakdown.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 0.28 * len(breakdown) + 2))
    ax.barh(breakdown["functionality"], breakdown["f1"].fillna(0), color="steelblue")
    ax.set_xlabel("F1")
    ax.set_title("HateCheck per-functionality F1")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(REPORTS / "hatecheck_breakdown.png", dpi=150)
    plt.close(fig)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "evaluation.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"reports written to {REPORTS}")


if __name__ == "__main__":
    main()
