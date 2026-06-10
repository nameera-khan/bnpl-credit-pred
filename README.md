# BNPL Credit Risk Scorer

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A production-style machine learning pipeline that predicts the probability of a customer defaulting on a Buy Now Pay Later (BNPL) instalment serving predictions via a REST API in under 100ms.

---

## The Business Problem

Buy Now Pay Later platforms approve or decline instalment plans at checkout in real time. A customer who defaults costs the platform the full loan amount. A model that scores default risk accurately and fast enough to sit inside a checkout flow directly reduces credit losses.

This project builds that scoring system end to end: from raw financial data through to a live API endpoint that returns a risk decision, a calibrated default probability, and an explanation of the top contributing factors.

---

## Results

| Metric | Value | Interpretation |
|---|---|---|
| AUC-ROC | 0.8979 | Model correctly ranks a defaulter above a repayer approx. 90% of the time |
| Gini coefficient | 0.7958 | Captures 80% of maximum possible separation between good and bad customers |
| Average Precision | 0.25 | 3× above the random baseline of 0.081 |
| API latency (p99) | < 100ms | Suitable for real-time checkout scoring |
| Top 20% capture rate | 76% | Reviewing the top 20% of highest-risk applicants intercepts 76% of all defaults |

The Gini coefficient of 0.79 places this model in the strong range for real credit data. Production bank scorecards typically target 0.40–0.65 on comparable datasets.

---

## Architecture

```
7 CSV files (2.7 GB)
    → SQLite database       (data/ingest.py)
    → Feature engineering   (data/features.py)
    → LightGBM training     (training/train.py)
    → Calibrated model      (model/lgbm_calibrated.pkl)
    → FastAPI serving       (api/main.py)
    → Drift monitoring      (monitoring/drift_detector.py)
```

The key design decision: all feature aggregations are computed once at training time and saved. At inference time the API assembles a flat feature vector from the request payload keeping latency consistently under 100ms.

---

## Project Structure

```
bnpl-credit-pred/
├── data/
│   ├── raw/                    # Home Credit CSV files (not committed)
│   ├── db/                     # SQLite database (not committed)
│   ├── ingest.py               # Loads CSVs into SQLite
│   └── features.py             # SQL aggregations → flat feature table
├── training/
│   └── train.py                # LightGBM + calibration + SHAP + MLflow
├── monitoring/
│   ├── drift_detector.py       # Evidently drift detection
│   ├── reference_data.parquet  # Training distribution snapshot
│   ├── shap_summary.png        # Feature importance plot
│   └── roc_lorenz_curves.png   # ROC and Lorenz curve plots
├── api/
│   └── main.py                 # FastAPI endpoints
├── model/                      # Saved pkl artifacts (not committed)
├── utils/
│   └── logging_config.py       # Structured logging
├── notebooks/
│   └── feature_engineering.ipynb
│   └── training.ipynb        # EDA and training plots
├── tests/
├── requirements.txt
└── Makefile
```

---

## Quickstart

**Prerequisites:** Conda, Kaggle account, Home Credit dataset accepted at kaggle.com/c/home-credit-default-risk/rules

```bash
# 1. Clone and enter the project
git clone https://github.com/YOUR_USERNAME/bnpl-credit-pred.git
cd bnpl-credit-pred

# 2. Create and activate the environment
conda create -n bnpl-credit-pred python=3.11 -y
conda activate bnpl-credit-pred
conda install -c conda-forge lightgbm scipy -y
pip install -r requirements.txt

# 3. Download the dataset
kaggle competitions download -c home-credit-default-risk -p data/raw/
cd data/raw && unzip home-credit-default-risk.zip && cd ../..

# 4. Load data into SQLite
python -m data.ingest

# 5. Train the model (~5 minutes)
python -m training.train

# 6. Start the API
uvicorn api.main:app --reload --port 8000
```

---

## API Endpoints

Once running, visit **http://localhost:8000/docs** for the full interactive documentation.

### POST /score

Score a single credit application and return a risk decision.

**Example request:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "amt_credit": 200000,
    "amt_annuity": 10000,
    "amt_income_total": 120000,
    "amt_goods_price": 180000,
    "age_years": 42,
    "years_employed": 8.5,
    "ext_source_1": 0.72,
    "ext_source_2": 0.81,
    "ext_source_3": 0.68,
    "education_encoded": 3,
    "bureau_loan_count": 2,
    "prev_approval_rate": 0.8,
    "install_late_count": 0
  }'
```

**Example response:**
```json
{
  "decision": "APPROVE",
  "default_probability": 0.0136,
  "risk_tier": "low",
  "credit_limit_factor": 1.0,
  "top_risk_factors": [
    {"feature": "ext_source_mean", "impact": -0.0842, "direction": "decreases_risk"},
    {"feature": "credit_income_ratio", "impact": 0.0312, "direction": "increases_risk"}
  ],
  "latency_ms": 68.6
}
```

| Decision | Condition | Action |
|---|---|---|
| APPROVE | probability < threshold × 0.7 | Proceed with instalment plan |
| MANUAL_REVIEW | probability between thresholds | Human review required |
| DECLINE | probability > threshold × 1.5 | Reject application |

### GET /health
Returns server status, model version, and predictions served count.

### GET /drift
Compares recent prediction features against the training distribution. Returns drift share and retraining recommendation after 100+ predictions.

---

## Model Details

### Dataset
Home Credit Default Risk: 307,511 loan applications across 7 relational tables. Default rate: 8.1% (severe class imbalance).

### Feature Engineering
All features are computed from SQL aggregations across the 7 tables and joined into a single flat row per customer. Key feature groups:

- **Application features** — credit-to-income ratio, annuity-to-income ratio, age, employment duration
- **External scores** — EXT_SOURCE_1/2/3 (credit bureau scores, strongest predictors)
- **Bureau history** — loan count, active loans, total debt, overdue count
- **Payment behaviour** — late payment count, average delay days
- **Prior applications** — approval rate, refused count

### Training Decisions

**Class imbalance:** `scale_pos_weight=11.3` — each defaulter weighted 11× more than a repayer during training. Without this correction the model predicts "repaid" for everyone and catches zero defaulters despite 92% accuracy.

**Probability calibration:** Raw LightGBM scores cluster near 0 and 1 and are overconfident. `CalibratedClassifierCV` with isotonic regression corrects this so a predicted probability of 35% means roughly 35% of such customers actually default.

**Decision threshold:** Optimised at 0.1764 (not the default 0.5) by maximising F1 on the precision-recall curve. Lower threshold reflects the asymmetric cost of missing a defaulter versus a false alarm.

**Temporal split:** Rows ordered by customer ID as a time proxy. First 80% used for training, last 20% for testing 

### Top Features by Importance

| Feature | Importance | Description |
|---|---|---|
| EXT_SOURCE_2 | ~15% | External credit bureau score 2 |
| EXT_SOURCE_3 | ~12% | External credit bureau score 3 |
| EXT_SOURCE_1 | ~10% | External credit bureau score 1 |
| credit_income_ratio | ~7% | Credit amount ÷ annual income |
| annuity_income_ratio | ~6% | Monthly payment ÷ annual income |
| age_years | ~5% | Applicant age |
| bureau_total_debt | ~5% | Total outstanding debt across all bureau loans |
| install_late_count | ~4% | Number of late instalment payments |

---

## Evaluation Plots

### ROC Curve and Lorenz / CAP Curve

The Lorenz curve shows the operational value directly: by reviewing the top 20% of applicants ranked by model score, the risk team intercepts 76% of all defaults compared to just 20% with random selection.

The Gini coefficient (0.7958) is the area between the model curve and the random diagonal, rescaled to [0, 1]. It is the standard discrimination metric used by credit risk teams and regulators.

### SHAP Feature Importance

SHAP values explain each individual prediction showing exactly which features pushed the default probability up or down for a specific applicant. This satisfies the explainability requirement for credit decisions under UAE CBUAE and international lending guidelines.

---

## Drift Monitoring

The `/drift` endpoint compares incoming application features against the training data distribution using Evidently. If more than 25% of features show statistically significant shift, it flags for retraining.

In a BNPL context, drift occurs when:
- Acquisition channels change (different customer profiles enter the system)
- Macroeconomic conditions shift (rising unemployment, interest rate changes)
- Seasonal patterns emerge (example: Ramadan spending behaviour in UAE)

---

## Skills Demonstrated

| Skill | Where |
|---|---|
| SQL — joins, aggregations, CASE statements | `data/features.py` |
| Clean Python — type hints, modules, logging | All modules |
| Project structure — not a notebook dump | Full repo layout |
| Git and version control | Commit history |
| APIs — FastAPI with Pydantic validation | `api/main.py` |
| Logging and monitoring | `utils/logging_config.py`, `monitoring/` |
| CI/CD — lint gate, test gate | `Makefile` |
| Cloud basics — S3-ready artifact loading | `api/main.py` lifespan |
| terminal — Makefile commands | `Makefile` |

---

## Failures Debugged

These are real issues encountered during the build 

**1. `cv="prefit"` removed in scikit-learn 1.4**
`CalibratedClassifierCV` no longer accepts the string `"prefit"`. Fixed by changing to `cv=None`, which correctly signals that the base estimator is already fitted.

**2. scipy version conflict**
Installing packages in the wrong order caused `scipy.spatial.transform._rotation` to have mismatched binary and Python files, breaking SHAP. Fixed by installing `scipy` via conda before pip touches anything — conda resolves binary dependencies that pip cannot.

**3. libomp missing on Mac**
LightGBM requires OpenMP (`libomp.dylib`) for parallel computation. pip installs LightGBM but not its system dependency. Fixed with `brew install libomp`.

**4. `ModuleNotFoundError: No module named 'utils'`**
Running `python data/ingest.py` sets the working directory to `data/` — Python cannot find the `utils/` sibling folder. Fixed by running all scripts as modules from the project root: `python -m data.ingest`.

---

## What a Production Version Would Add

- **Feature store** — pre-compute and cache customer features on a schedule rather than at inference time
- **Airflow/Prefect** — scheduled retraining pipeline triggered by drift alerts
- **A/B testing** — shadow mode deployment to compare new model against current before full rollout
- **Database prediction logging** — replace in-memory list with PostgreSQL for persistent audit trail
- **Authentication** — API key or OAuth on all endpoints
- **Horizontal scaling** — multiple API replicas behind a load balancer for high throughput

---

## Running with Docker

```bash
docker build -t bnpl-scorer .
docker run -p 8000:8000 bnpl-scorer
```

---

## Makefile Commands

```bash
make install    # install all dependencies
make ingest     # load CSVs into SQLite
make train      # train and save the model
make serve      # start the API on port 8000
make test       # run pytest
make lint       # ruff + black checks
make format     # auto-fix formatting
```

---

## Dataset

Home Credit Default Risk — available at kaggle.com/c/home-credit-default-risk

You must accept the competition rules on Kaggle before the CLI download will work. The dataset is not committed to this repository.

---

## License

MIT
