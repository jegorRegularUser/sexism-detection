"""Fine-tune DistilBERT for binary sexism detection.

Trains on data/processed/{train,val}.csv, evaluates each epoch on the
stratified validation split, keeps the best checkpoint by macro-F1 with
early stopping (patience=2), then saves the final model + tokenizer and a
small report (metrics JSON + training curves PNG).

Class imbalance is handled with inverse-frequency class weights in the
cross-entropy loss instead of oversampling (keeps the data distribution
honest and works better with a stratified split).

Run:  python ml/train.py --model-name distilbert-base-uncased
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
import transformers
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


class SexismDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_attention_mask=True,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": self.labels[idx],
        }


class WeightedTrainer(Trainer):
    """Cross-entropy with inverse-frequency class weights."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = probs.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs[:, 1]),
    }


class PadCollator:
    """Dynamic padding to the longest sequence in a batch.

    A picklable class instead of a closure: dataloader workers on Windows
    use spawn and cannot pickle local functions.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, attention, labels = [], [], []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            attention.append(b["attention_mask"] + [0] * pad)
            labels.append(b["labels"])
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention),
            "labels": torch.tensor(labels),
        }


def class_weights(labels: list[int]) -> torch.Tensor:
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    weights = counts.sum() / (2.0 * counts)
    print(f"class counts: {counts.tolist()} -> weights: {weights.round(3).tolist()}")
    return torch.tensor(weights, dtype=torch.float32)


def plot_curves(log_history: list[dict], out_png: Path) -> None:
    steps = [h for h in log_history if "eval_f1" in h]
    epochs = [h["epoch"] for h in steps]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["eval_f1"] for h in steps], "o-", label="eval f1")
    axes[0].plot(epochs, [h["eval_roc_auc"] for h in steps], "s-", label="eval roc_auc")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].set_title("Validation metrics")
    train_steps = [h for h in log_history if "loss" in h and "epoch" in h]
    axes[1].plot([h["epoch"] for h in train_steps], [h["loss"] for h in train_steps], "-")
    axes[1].set_xlabel("epoch")
    axes[1].set_title("Training loss")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        default=str(ROOT / "models" / "sexism-distilbert-v2"),
    )
    args = p.parse_args()

    transformers.set_seed(args.seed)
    REPORTS.mkdir(exist_ok=True)
    out_dir = Path(args.out)

    train_df = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    val_df = pd.read_csv(ROOT / "data" / "processed" / "val.csv")
    print(f"train={len(train_df)} (pos {train_df.label.sum()}), "
          f"val={len(val_df)} (pos {val_df.label.sum()})")

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    train_ds = SexismDataset(
        train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, args.max_len
    )
    val_ds = SexismDataset(
        val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, args.max_len
    )

    # transformers 5.x dropped warmup_ratio; pass an absolute step count
    # (10% of the schedule) instead.
    total_steps = args.epochs * (len(train_ds) + args.batch_size - 1) // args.batch_size
    warmup_steps = max(1, int(0.1 * total_steps))

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=1,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = WeightedTrainer(
        model=transformers.AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=2
        ),
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights(train_df["label"].tolist()),
    )

    trainer.train()

    final_val = trainer.evaluate()
    print("final validation:", final_val)

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Deployed decision threshold: argmax-F1 on validation, shipped with the
    # model so that back.py / predict.py / evaluate.py all agree on it.
    val_out = trainer.predict(val_ds)
    val_probs = torch.softmax(torch.tensor(val_out.predictions), dim=-1)[:, 1].numpy()
    val_labels = val_df["label"].to_numpy()
    sweep = [
        (float(t), float(f1_score(val_labels, (val_probs >= t).astype(int))))
        for t in np.arange(0.05, 0.96, 0.01)
    ]
    best_threshold, best_f1_at_t = max(sweep, key=lambda x: x[1])
    (out_dir / "threshold.json").write_text(
        json.dumps({"threshold": best_threshold, "selection": "argmax F1 on validation"},
                   indent=2),
        encoding="utf-8",
    )
    print(f"deployed decision threshold: {best_threshold:.2f} (val F1 {best_f1_at_t:.3f})")

    metrics = {
        "model_name": args.model_name,
        "epochs_requested": args.epochs,
        "epochs_run": trainer.state.epoch,
        "decision_threshold": best_threshold,
        "args": {k: getattr(args, k) for k in vars(args)},
        "validation": {k: float(v) for k, v in final_val.items()},
    }
    (REPORTS / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (REPORTS / "training_log.json").write_text(
        json.dumps(trainer.state.log_history, indent=2), encoding="utf-8"
    )
    plot_curves(trainer.state.log_history, REPORTS / "training_curves.png")
    print(f"model saved to {out_dir}, reports in {REPORTS}")


if __name__ == "__main__":
    main()
