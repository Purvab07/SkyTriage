"""Scoring for the ATA classifier.

Accuracy alone is misleading here: ATA 53 (fuselage) is about 38% of reports, so a
model that always says "53" already scores ~38% accuracy. Macro-F1 averages the score
of every chapter equally, so rare systems count as much as common ones.

Run on a saved model:
    python -m eval.eval_classifier models/ata_baseline_<version>.joblib
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from ml.datasets import LABEL_COL, TEXT_COL, ata_name, load_sdr, time_split


def majority_baseline(y_train: Sequence[str], n: int) -> list[str]:
    """The 'no intelligence' model: always predict the most common training label."""
    most_common = Counter(y_train).most_common(1)[0][0]
    return [most_common] * n


def evaluate(y_true: Sequence[str], y_pred: Sequence[str], top_confusions: int = 10) -> dict:
    y_true, y_pred = list(y_true), list(y_pred)
    labels = sorted(set(y_true))  # score only chapters that appear in the true labels
    per_class = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pairs = [
        (labels[i], labels[j], int(cm[i, j]))
        for i in range(len(labels))
        for j in range(len(labels))
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(key=lambda p: p[2], reverse=True)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            lab: {
                "precision": per_class[lab]["precision"],
                "recall": per_class[lab]["recall"],
                "f1": per_class[lab]["f1-score"],
                "support": int(per_class[lab]["support"]),
            }
            for lab in labels
        },
        "top_confusions": pairs[:top_confusions],
    }


def write_report(metrics: dict, naive: dict, meta: dict, path: Path) -> Path:
    """Write a Markdown report you can paste into the README."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# ATA classifier report: {meta.get('model', 'model')} ({meta.get('version', '')})",
        "",
        f"Data: {meta.get('splits', '')}",
        "",
        "| Model | Macro-F1 | Weighted-F1 | Accuracy |",
        "| --- | --- | --- | --- |",
        (
            f"| Always majority class | {naive['macro_f1']:.3f} | {naive['weighted_f1']:.3f} "
            f"| {naive['accuracy']:.3f} |"
        ),
        (
            f"| {meta.get('model', 'model')} | {metrics['macro_f1']:.3f} | "
            f"{metrics['weighted_f1']:.3f} | {metrics['accuracy']:.3f} |"
        ),
        "",
        "## Per chapter (sorted by support)",
        "",
        "| Chapter | Name | Support | Precision | Recall | F1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    rows = sorted(metrics["per_class"].items(), key=lambda kv: kv[1]["support"], reverse=True)
    for lab, m in rows:
        lines.append(
            f"| {lab} | {ata_name(lab)} | {m['support']} | {m['precision']:.2f} "
            f"| {m['recall']:.2f} | {m['f1']:.2f} |"
        )
    lines += ["", "## Most common mix-ups (true -> predicted)", ""]
    lines += [
        f"- {t} {ata_name(t)} -> {p} {ata_name(p)}: {n}" for t, p, n in metrics["top_confusions"]
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score a saved ATA classifier on the test split.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    splits = time_split(load_sdr())
    model = joblib.load(args.model_path)
    y_test = splits.test[LABEL_COL]
    y_trval = pd.concat([splits.train[LABEL_COL], splits.val[LABEL_COL]])
    metrics = evaluate(y_test, model.predict(splits.test[TEXT_COL]))
    naive = evaluate(y_test, majority_baseline(list(y_trval), len(y_test)))
    out = args.out or args.model_path.with_suffix(".report.md")
    meta = {"model": args.model_path.stem, "splits": splits.summary()}
    print(f"macro-F1 {metrics['macro_f1']:.3f}  accuracy {metrics['accuracy']:.3f}")
    print(f"Report: {write_report(metrics, naive, meta, out)}")


if __name__ == "__main__":
    main()