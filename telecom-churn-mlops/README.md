# Telecom Customer Churn — Production ML System

Production version of [Telecom Customer Churn Prediction: Ensemble ML & Threshold Optimisation](https://github.com/sMathujan/Telecom-Customer-Churn-Prediction-Ensemble-ML-Threshold-Optimisation). It turns the three research notebooks into tracked, reproducible pipelines and a REST API, and it follows the structure of the [BPRMLS template](https://github.com/sMathujan/Template-for-BPRMLS).

Random Forest, XGBoost and CatBoost are tuned with Optuna, using SMOTE inside each CV fold. Each model gets a decision threshold chosen on **validation** data, and the champion is served behind FastAPI with MLflow tracking every run.

---

## 🎯 What the system does

```
raw CSV ──► Data pipeline ──► Training pipeline ──► champion model + frozen threshold ──► Inference
             clean               Optuna (CV, SMOTE-in-fold)      artifacts/models/            stream / batch / REST API
             split 60/20/20      refit on SMOTE train            MLflow registry @champion    MLflow monitoring
             fit features on     threshold ← validation
             TRAIN only          evaluate once on test
```

The core idea from the research is preserved end to end. A churn model outputs a probability, and **the threshold that turns it into "send a retention offer" is a business decision**. That threshold is learned on validation data, saved alongside the model and applied unchanged in production.

---

## 📁 Project structure

```
telecom-churn-mlops/
├── Makefile                      # One-command automation (make help)
├── config.yaml                   # Single source of truth for every setting
├── requirements.txt              # Full dev environment
├── requirements-serve.txt        # Minimal runtime for Docker
├── Dockerfile / .dockerignore    # Serving image (API + trained artifacts)
├── .github/workflows/ci.yml      # Lint → build data → baseline model → tests on every push
├── .flake8 / pytest.ini
│
├── data/
│   ├── raw/Dataset-Telco-Customer-Churn.csv   # IBM Telco (7,043 customers)
│   └── processed/                              # Cleaned data (generated)
│
├── pipelines/
│   ├── data_pipeline.py                 # Ingest → clean → split → features (fit on train)
│   ├── training_pipeline.py             # Tune → refit → threshold → evaluate → champion
│   └── streaming_inference_pipeline.py  # Single / stream / batch scoring + monitoring
│
├── src/
│   ├── data_ingestion.py         # CSV / Excel ingestors with schema validation
│   ├── data_cleaning.py          # TotalCharges coercion, service-category normalisation
│   ├── handle_missing_values.py  # Drop (training) / fill (inference) strategies
│   ├── outlier_detection.py      # IQR detection (report-only by default)
│   ├── feature_binning.py        # Optional tenure bins (off by default)
│   ├── feature_encoding.py       # Binary maps + one-hot (drop_first), persisted encoders
│   ├── feature_scaling.py        # StandardScaler fitted on TRAIN only, persisted
│   ├── preprocessing.py          # ChurnPreprocessor: ONE pipeline for training & serving
│   ├── data_splitter.py          # Stratified 60/20/20 split with integrity checks
│   ├── class_imbalance.py        # SMOTE (train split only)
│   ├── model_building.py         # RF / XGBoost / CatBoost builders
│   ├── hyperparameter_tuning.py  # Optuna TPE, SMOTE inside each CV fold
│   ├── threshold_optimisation.py # F1-optimal or £-expected-value thresholds
│   ├── model_training.py         # Training + model/metadata persistence
│   ├── model_evaluation.py       # Metrics at frozen threshold, bootstrap CI, plots
│   └── model_inference.py        # Raw record(s) → probability → decision
│
├── api/
│   ├── main.py                   # FastAPI: /health /model-info /predict /predict/batch /reload
│   └── schemas.py                # Pydantic schemas (invalid categories → HTTP 422)
│
├── utils/
│   ├── config.py                 # Config access + project-root path resolution
│   ├── logger.py                 # Consistent logging
│   └── mlflow_utils.py           # Runs, metrics, model registry + @champion alias
│
├── tests/                        # 16 tests: unit + inference + API
│
└── artifacts/                    # Generated: data splits, encoders, scaler, model, plots
```

---

## 🚀 Quick start

```bash
git clone <this-repo> && cd telecom-churn-mlops
make install && source .venv/bin/activate

make run-all            # data → quick training (5 trials/model) → sample prediction
make test               # 16 tests
make serve              # API at http://localhost:8000/docs
make mlflow-ui          # experiments at http://localhost:5001
```

For research-grade results (50 / 100 / 100 Optuna trials, as in Notebook 03), run `make train-pipeline`. This takes a while.

| Command | What it does |
|---|---|
| `make data-pipeline` | Builds splits/encoders/scaler (reuses if present; `data-pipeline-rebuild` forces) |
| `make train-pipeline` | Full Optuna tuning of all three models, champion promoted |
| `make train-pipeline-fast TRIALS=10` | Same, fewer trials |
| `make train-baseline` | No tuning — the research baseline hyperparameters |
| `make streaming-inference` | Scores one sample customer |
| `make stream-demo` | Replays 25 customers as a stream with MLflow monitoring |
| `make batch-inference` | Scores a whole CSV → `artifacts/predictions/predictions.csv` |
| `make docker-build && make docker-run` | Containerised API (train first) |

Training options can also be passed directly:

```bash
python pipelines/training_pipeline.py --models xgboost random_forest --n-trials 20
python pipelines/training_pipeline.py --no-tune --force-data
```

---

## 🔬 Methodology (faithful to the research)

| Step | Implementation | Why |
|---|---|---|
| Clean | Blank `TotalCharges` (11 tenure-0 rows) dropped; "No internet/phone service" → "No" | Notebook 01 |
| Split first | Stratified 60 / 20 / 20 (4,219 / 1,406 / 1,407, 26.6% churn each) | Threshold is a learned parameter; test must stay untouched |
| Features | Binary maps, one-hot `drop_first`, StandardScaler, **all fitted on train only** | No leakage into validation/test/production |
| Balance | SMOTE on training split only; re-applied inside each CV fold during tuning | Synthetic samples never leak into evaluation |
| Tune | Optuna TPE, 5-fold stratified CV F1 on raw train | Notebook 03 search spaces reproduced exactly |
| Refit | On SMOTE-balanced train; XGB/CatBoost use the mean CV early-stopping round | Validation reserved for threshold choice |
| Threshold | Scanned 0.10–0.89 on **validation** | Never chosen on test |
| Select | Champion = best **validation** F1 | Test is opened once, for reporting |
| Report | Test metrics at frozen threshold + bootstrap 95% CI + £ expected value | Honest, comparable numbers |

The production data pipeline reproduces the research exactly: the same 11 rows are dropped, the split sizes are identical and the **23 model features match `processed_telco_scaled.csv` column for column**.

### Differences from the notebooks (deliberate fixes)

1. **Scaler leakage removed.** Notebook 01 fits `StandardScaler` on the full dataset before splitting. Here it is fitted on the training split only.
2. **Champion chosen on validation.** Notebook 03 selected the best model by *test* F1, while the research README states model selection happens on validation. The pipeline now does what the README describes.
3. **No training/serving skew.** The template re-implemented preprocessing inside `model_inference.py`. Here, a single `ChurnPreprocessor` is fitted once, saved, and loaded for inference.

Because of (1), metrics differ slightly from the published table, mainly through SMOTE, which is distance-based and so sensitive to scaling.

---

## 📈 Results (pipeline run, 15 Optuna trials per model)

Test set of 1,407 customers, opened once. Each threshold was chosen on validation.

| Model | Threshold | Recall | Precision | F1 (95% CI) | ROC-AUC | PR-AUC | £ value |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.49 | 77.3% | 54.3% | 0.638 (0.600–0.673) | 0.845 | 0.636 | £90,907 |
| **XGBoost** ⭐ | 0.57 | 85.6% | 51.0% | 0.639 (0.603–0.673) | 0.851 | 0.650 | £96,903 |
| CatBoost | 0.57 | 75.9% | 54.4% | 0.634 (0.597–0.670) | 0.849 | 0.659 | £89,413 |

⭐ Champion, chosen on validation F1 (0.622).

These numbers match the research conclusion: **the models are not meaningfully different**, because every confidence interval overlaps. The full 50/100/100-trial run (`make train-pipeline`) is the research-grade configuration and will move these slightly.

The threshold story also holds. XGBoost's F1-optimal threshold (0.57) gives a higher F1 than 0.50 (0.639 vs 0.621) but *less* money (£96,903 vs £97,133). Switching to `metric: "expected_value"` picks 0.56 and earns £97,405, so optimising for the business objective beats both.

---

## 💷 Business-aware thresholds

`config.yaml` holds the UK retention economics from the research:

```yaml
threshold_optimisation:
  metric: "f1"              # or "expected_value"

business:
  customer_lifetime_value_gbp: 1656.0   # £69/month × 24 months (Ofcom)
  retention_offer_cost_gbp: 99.0        # one year of a typical discount (Which?)
  retention_success_rate: 0.30          # assumption
```

Every model is evaluated in pounds as well as in F1:

```
expected value = TP × (save_rate × CLV) − (TP + FP) × offer_cost
```

The test suite checks that this formula reproduces the README's £79,249 (threshold 0.50) and £88,650 (threshold 0.37) exactly. Setting `metric: "expected_value"` makes the pipeline pick the threshold that maximises retention revenue instead of F1. This is the research's main conclusion turned into a config switch.

---

## 🌐 REST API

```bash
make serve
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
  "Dependents": "No", "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
  "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
  "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85}'
```

```json
{"customerID": "7590-VHVEG", "churn_probability": 0.8254, "prediction": 1, "status": "Churn",
 "risk_band": "High", "recommended_action": "Send retention offer", "threshold": 0.57,
 "model_name": "XGBoost"}
```

The endpoints are listed below; interactive docs are at `/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness and whether a model is loaded |
| `GET /model-info` | Champion name, version, threshold, test metrics, CI, economics |
| `POST /predict` | Score one customer (raw schema, same as the source CSV) |
| `POST /predict/batch` | Score up to 10,000 customers |
| `POST /reload` | Hot-reload after retraining |

Unknown categories (e.g. `"Contract": "Three year"`) are rejected with HTTP 422. `prediction` comes from the frozen threshold. `risk_band` is a separate probability band for prioritisation (`config.yaml → inference.risk_bands`).

---

## 📊 MLflow tracking

MLflow uses a local SQLite backend (`mlflow.db`) with artifacts in `mlruns/`. Recent MLflow releases no longer accept the plain `./mlruns` file store that the template used. Set `MLFLOW_TRACKING_URI` to point at a remote server instead.

| Run | Logged |
|---|---|
| `data pipeline` | Stage row/missing counts, outlier report, split sizes & churn rates, dataset lineage, EDA plots, encoders/scaler |
| `training pipeline` (parent) | Comparison table + chart, champion metadata, registered model `telecom_churn_model@champion` |
| ↳ one child run per model | Params, CV F1, Optuna history, threshold, val/test metrics, test@0.50, bootstrap CI, £ value, ROC/PR, threshold sweep, confusion matrix, feature importance |
| `streaming inference` | Per-batch latency (avg/p95), mean probability, flag rate, risk-band counts, raw batches |

---

## ✅ Testing

```bash
make test
```

- **Unit tests:** cleaning, encoding parity with `pd.get_dummies`, encoder save/load, stratified split integrity, threshold search, and README £ figures.
- **Integration tests:** inference, blank `TotalCharges`, column order matching training, missing-field errors, and all API endpoints.

Integration tests skip automatically if no model has been trained.

---

## 🔭 Next steps

- Data-drift monitoring (e.g. PSI on `tenure`, `MonthlyCharges`, `Contract` mix) against the training distribution
- Scheduled retraining with automatic promotion only if validation F1 / £ value improves
- Probability calibration (isotonic) so `risk_band` reflects true churn likelihood
- SHAP explanations per prediction for retention-team transparency
- Repeated-split evaluation (the research's 30-split experiment) as a pipeline stage

---

## Data

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). This is a fictional telecommunications company dataset from IBM Sample Datasets.

## Author

**Mathujan Sivananthan**, MSc Data Science, University of Hertfordshire
[LinkedIn](https://linkedin.com/in/mathujan-sivananthan) · [GitHub](https://github.com/sMathujan)
