"""CLI inference demo for the trained sexism detector.

Run:  python ml/predict.py "your text here"
      echo "text" | python ml/predict.py
      python ml/predict.py --model models/sexism-distilbert-v2 --text-file input.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "sexism-distilbert-v2"


def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    threshold = 0.5
    meta = Path(model_dir) / "threshold.json"
    if meta.exists():
        threshold = float(json.loads(meta.read_text(encoding="utf-8"))["threshold"])
    return tokenizer, model.to(device), device, threshold


@torch.no_grad()
def predict(texts: list[str], tokenizer, model, max_len: int = 128,
            threshold: float = 0.5) -> list[dict]:
    enc = tokenizer(texts, truncation=True, max_length=max_len, padding=True,
                    return_tensors="pt").to(model.device)
    probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().tolist()
    return [
        {"text": t, "label": int(p >= threshold), "sexist_probability": round(p, 4)}
        for t, p in zip(texts, probs)
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("texts", nargs="*", help="texts to classify")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--text-file", help="one text per line")
    args = p.parse_args()

    texts = list(args.texts)
    if args.text_file:
        texts += Path(args.text_file).read_text(encoding="utf-8").splitlines()
    if not texts and not sys.stdin.isatty():
        texts += [line.rstrip("\n") for line in sys.stdin]
    if not texts:
        p.error("no input texts (pass as args, --text-file, or stdin)")

    tokenizer, model, device, threshold = load_model(args.model)
    print(f"model: {args.model} on {device}, threshold={threshold}")
    for row in predict(texts, tokenizer, model, threshold=threshold):
        verdict = "SEXIST" if row["label"] else "not sexist"
        print(f"[{verdict:>10}] p={row['sexist_probability']:.3f}  {row['text'][:100]}")


if __name__ == "__main__":
    main()
