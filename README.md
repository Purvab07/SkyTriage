# SkyTriage

**AI-assisted defect triage for aviation maintenance.** Paste a mechanic's defect write-up and SkyTriage predicts which aircraft system it belongs to, with a confidence score you can trust and a short list of alternatives when it is unsure.

Built on 185,520 real FAA Service Difficulty Reports (2024 to 2026).

> **Advisory only.** SkyTriage never replaces the official Airworthiness Directive, approved maintenance data, or a certified mechanic's judgement.

<!-- Add a screenshot of the demo page here: ![SkyTriage demo](docs/demo.png) -->

---

## Why this project

Maintenance teams describe defects in free text full of shorthand ("MLG ACTR LKG AT FWD FITTING"). Turning that into structured, searchable knowledge is slow and manual. SkyTriage is a decision-support system that will combine:

- **Classification:** which aircraft system (ATA chapter) is affected
- **Calibrated uncertainty:** how far to trust each prediction, and what else it could be
- **Similar-case retrieval:** has this happened before, and what fixed it *(in progress)*
- **AD compliance checks:** which Airworthiness Directives may apply to a specific aircraft *(planned)*
- **Trend detection:** which problems are becoming more common *(planned)*
- **Verified explanations:** every flag explained in plain English, checked against its evidence *(planned)*

## Results so far

All numbers are on **2026 reports the model never saw** (time-based split: train on Jan 2024 to Jun 2025, tune on Jul to Dec 2025, test on Jan to Nov 2026).

### Classification (51,788 test reports, 50+ aircraft systems)

| Model | Macro-F1 | Accuracy |
| --- | --- | --- |
| Always predict the most common system | 0.011 | 36.5% |
| **TF-IDF (word + character n-grams) + logistic regression** | **0.624** | **91.4%** |

Macro-F1 weights every aircraft system equally, so it exposes weaker performance on rare systems that accuracy hides. Fuselage alone is 38% of reports.

### Calibration

| | Expected Calibration Error (lower is better) |
| --- | --- |
| Raw model | 0.062 |
| **After temperature scaling (T = 2.32)** | **0.017** |

The raw model was overconfident. Temperature scaling cut calibration error by 73% without changing a single prediction, so "80% sure" now means right about 80% of the time.

### Conformal prediction sets

| Coverage target | Coverage on 2026 data | Average set size |
| --- | --- | --- |
| 90% | 90.9% | 1.00 |
| **95%** | **94.3%** | **1.12** |

At 90% the sets collapsed to single answers because accuracy already exceeded the target, so the system uses 95%: most reports get one confident answer, and uncertain ones get a short "could also be" list. The small shortfall against 95% reflects drift between 2025 and 2026 reports and is handled by periodic recalibration.

## Try it

```bash
# 1. Setup (Python 3.12, Docker Desktop running)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ml]"
cp .env.example .env
docker compose up -d db redis

# 2. Data: download yearly SDR CSVs from https://www.faa.gov/av-info/download_SDR
#    into data/raw/ as SDR-2024.csv, SDR-2025.csv, ...
python -m pipelines.ingest_sdr

# 3. Train
python -m ml.train_baseline
python -m ml.train_calibrated --conformal-alpha 0.05

# 4. Run
uvicorn app.main:app --reload
```

- Demo page: http://localhost:8000/demo
- Interactive API docs: http://localhost:8000/docs

### API example

```bash
curl -X POST http://localhost:8000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "MLG ACTUATOR LEAKING HYD FLUID. REPLACED SEAL."}'
```

```json
{
  "top": {"chapter": "32", "name": "Landing Gear", "probability": 0.93},
  "prediction_set": [{"chapter": "32", "name": "Landing Gear", "probability": 0.93}],
  "coverage_target": 0.95,
  "confident": true,
  "model_version": "20261114-021636",
  "advisory": "Advisory only. Confirm with the official AD, approved maintenance data and a certified mechanic before acting."
}
```

`confident` is true only when there is a single candidate and its calibrated probability clears the cascade threshold (0.85). Later, low-confidence cases will be routed to deeper analysis instead of the fast path.

## How it works

```
FAA SDR CSVs ─► ingest + clean + validate ─► Postgres (185,520 reports) + Parquet
                                                    │
                                         time-based split (no future leakage)
                                                    │
                      TF-IDF + logistic regression ─► temperature scaling ─► conformal sets
                                                    │
                                     versioned model bundle (models/*.joblib + metadata)
                                                    │
                       FastAPI  POST /v1/classify ─► demo page / API clients
```

## Key design decisions

- **Time-based evaluation, not random splits.** In real use the model always sees future reports. A random split lets it peek at reports from the same week and inflates scores.
- **Each data slice has one job.** Training data fits the model, the first half of the tuning period fits the temperature, the second half fits the conformal threshold, and 2026 data is used once for final scores.
- **Features vs. keys.** Serial and tail numbers are never model inputs (that would memorize aircraft instead of learning patterns). They are kept as keys for AD effectivity and repeat-defect detection. Part name and condition are kept out of the classifier because they leak the label.
- **Bad values are blanked, not dropped.** 80 reports claim impossible flight hours; the value is cleared and logged, but the rest of each report is kept.
- **Idempotent ingestion.** Re-running the pipeline never creates duplicates.
- **Portable model artifacts.** The model bundle class lives in its own module, so saved models load in any process. A regression test trains from the command line and loads in another process to keep it that way.
- **Fail clearly.** A missing or unreadable model returns a 503 with instructions, not a crash.

More detail: [`docs/data_notes.md`](docs/data_notes.md).

## Data

FAA Service Difficulty Reports, public yearly downloads. About 96% of reports concern transport-category aircraft (Boeing, Airbus, Bombardier, Embraer), so the system targets regional and charter operators first; similar-case search is thin for light training aircraft.

## Testing

```bash
pytest                      # 55 tests
pytest --cov=app --cov=pipelines --cov=ml --cov=eval --cov-report=term-missing
```

Covers cleaning and validation rules, cross-file de-duplication, time-split leakage, calibration and conformal maths, end-to-end training on a small synthetic dataset, the API (success, low confidence, invalid input, missing model), and an idempotent database load against a throwaway table. Database tests skip automatically when Postgres is not running.

## Project structure

```
app/         FastAPI service: routes, schemas, services
pipelines/   Data ingestion and cleaning
ml/          Training, calibration, conformal prediction, model bundle
eval/        Evaluation and reports (eval/reports/ holds every run's results)
tests/       Unit and integration tests
docs/        Data notes and design decisions
```

## Roadmap

- [x] SDR ingestion, cleaning and validation
- [x] Baseline classifier with time-based evaluation
- [x] Temperature-scaling calibration and conformal prediction sets
- [x] `/v1/classify` API and demo page
- [ ] Shorthand normalization, measured against the baseline
- [ ] Similar-case search (hybrid keyword + embedding retrieval, effectivity filter)
- [ ] AD ingestion and effectivity rule parsing
- [ ] Full `/v1/triage` with "why flagged" evidence and verified LLM explanations
- [ ] Fleet trend detection (clustering + change-point detection)
- [ ] Cost-aware model cascade, monitoring, CI evaluation gates
- [ ] Web UI for operators