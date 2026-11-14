"""Train the baseline ATA-chapter classifier: TF-IDF + linear model.

Text features:
  - word 1-2 grams ("landing gear", "selector valve")
  - character 3-5 grams, which cope with mechanic shorthand and typos
    ("RETR", "ACTR", "LKG" share pieces with "retract", "actuator", "leaking")
Model:
  - SGDClassifier with log loss = logistic regression trained quickly; it gives
    probabilities, which the cascade (#9) and calibration (#4) need later.
  - class_weight="balanced" so rare systems are not ignored in favour of ATA 53.

Run from the project folder:
    python -m ml.train_baseline
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, make_union

from eval.eval_classifier import evaluate, majority_baseline, write_report
from ml.datasets import LABEL_COL, TEXT_COL, load_sdr, time_split

log = logging.getLogger("train_baseline")

MODEL_DIR = Path("models")
REPORT_DIR = Path("eval/reports")


def build_pipeline(alpha: float = 2e-6, seed: int = 0) -> Pipeline:
    words = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=3, max_features=150_000,
        sublinear_tf=True,
    )
    chars = TfidfVectorizer(
        lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=5,
        max_features=150_000, sublinear_tf=True,
    )
    clf = SGDClassifier(
        loss="log_loss", alpha=alpha, class_weight="balanced",
        max_iter=60, tol=1e-4, random_state=seed, n_jobs=-1,
    )
    return Pipeline([("features", make_union(words, chars)), ("clf", clf)])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the baseline ATA classifier.")
    parser.add_argument("--alphas", type=float, nargs="+", default=[2e-7, 5e-7, 1e-6],
                        help="Regularisation strengths to try; the best on val is kept.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    splits = time_split(load_sdr())
    log.info("Splits: %s", splits.summary())
    X_train, y_train = splits.train[TEXT_COL], splits.train[LABEL_COL]
    X_val, y_val = splits.val[TEXT_COL], splits.val[LABEL_COL]

    # 1. Pick alpha on the validation set (never on test).
    best_alpha, best_f1 = None, -1.0
    for alpha in args.alphas:
        started = time.perf_counter()
        model = build_pipeline(alpha).fit(X_train, y_train)
        f1 = evaluate(y_val, model.predict(X_val))["macro_f1"]
        log.info("alpha=%g  val macro-F1=%.3f  (%.0fs)", alpha, f1, time.perf_counter() - started)
        if f1 > best_f1:
            best_alpha, best_f1 = alpha, f1

    # 2. Refit on train + val with the chosen alpha, then score once on test.
    started = time.perf_counter()
    X_trval = list(X_train) + list(X_val)
    y_trval = list(y_train) + list(y_val)
    final = build_pipeline(best_alpha).fit(X_trval, y_trval)
    train_seconds = time.perf_counter() - started

    y_test = splits.test[LABEL_COL]
    test_metrics = evaluate(y_test, final.predict(splits.test[TEXT_COL]))
    naive_metrics = evaluate(y_test, majority_baseline(y_trval, len(y_test)))
    log.info(
        "TEST  baseline macro-F1=%.3f acc=%.3f | always-majority macro-F1=%.3f acc=%.3f",
        test_metrics["macro_f1"], test_metrics["accuracy"],
        naive_metrics["macro_f1"], naive_metrics["accuracy"],
    )

    # 3. Save the model with metadata describing exactly how it was made.
    version = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f"ata_baseline_{version}.joblib"
    joblib.dump(final, model_path)
    meta = {
        "model": "tfidf_word12_char35_sgd_logloss_balanced",
        "version": version,
        "alpha": best_alpha,
        "val_macro_f1": best_f1,
        "train_rows": len(y_trval),
        "test_rows": len(y_test),
        "splits": splits.summary(),
        "train_seconds": round(train_seconds, 1),
        "test": {k: test_metrics[k] for k in ("macro_f1", "weighted_f1", "accuracy")},
        "always_majority": {k: naive_metrics[k] for k in ("macro_f1", "accuracy")},
    }
    model_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    report = write_report(test_metrics, naive_metrics, meta, REPORT_DIR / f"baseline_{version}.md")
    log.info("Saved model to %s and report to %s", model_path, report)


if __name__ == "__main__":
    main()