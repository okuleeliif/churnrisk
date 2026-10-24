# 📊 ChurnLens — SaaS Customer Churn Prediction & Decision Support Platform

ChurnLens is a Streamlit application that analyzes a SaaS company's customer
portfolio, predicts which customers are at risk of churning using machine
learning, and provides actionable **What-If simulations** ("what happens to
the risk if this customer's support tickets go down?") to support retention
decisions.

> The app interface is in Turkish.

## ✨ Features

- **Portfolio-wide risk scoring** — scores the entire customer base and shows
  the monthly recurring revenue (MRR) at risk.
- **Automatic segmentation** — buckets at-risk customers into 4 operational
  clusters by root cause (onboarding, engagement drop, support crisis,
  contract/payment risk), each downloadable as CSV.
- **Customer search & sorting** — find a customer by ID and sort the list by
  highest risk, lowest risk or monthly revenue.
- **What-If simulation** — instantly see how risk changes when you adjust a
  customer's attributes (contract type, support tickets, usage frequency,
  etc.), with a one-click reset for each field.
- **SHAP explainability** — visualizes why the model made a given prediction,
  broken down by feature contribution.
- **Automated intervention report** — generates a downloadable "rescue plan"
  report for the selected customer.
- **Pluggable data layer** — runs on a local CSV by default and can connect to
  Postgres/MySQL/SQLite or BigQuery through environment variables, with no
  code changes (see [Switching Data Sources](#️-switching-data-sources)).

## 🏗️ Architecture

```
generatedata.py  →  data/saas_churn_data.csv  →  data_layer/  →  trainmodel.py  →  models/*.pkl
                                                       │
                                                       ▼
                                                    app.py (Streamlit UI)
```

The `data_layer/` package decouples the data source (CSV / SQL / BigQuery)
from the rest of the application. Which source is used is determined by the
`CHURNLENS_DATA_SOURCE` environment variable.

## 📈 Data & Model

- **Synthetic data:** 5,000 SaaS customers with realistically correlated
  features, a noisy churn label, an imbalanced class distribution (~14% churn)
  and missing values in `feature_adoption_rate` and `nps_score`.
- **Model:** scikit-learn `HistGradientBoostingClassifier` (handles missing
  values natively), tuned with `RandomizedSearchCV` and 5-fold
  `StratifiedKFold`, using `class_weight="balanced"` for the imbalance.
- **Evaluation (held-out 20% test set):** ROC-AUC ≈ 0.78, PR-AUC ≈ 0.42
  (vs. a 0.14 baseline), churn recall ≈ 0.69. PR-AUC and the confusion matrix
  are reported alongside ROC-AUC because they are more informative on
  imbalanced data.

## 🚀 Setup

Requires Python 3.10+.

```bash
# 1. Clone the repository
git clone https://github.com/okuleeliif/churnrisk.git
cd churnrisk

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the demo dataset  (creates data/saas_churn_data.csv)
python generatedata.py

# 5. Train the model  (creates the files in models/)
python trainmodel.py

# 6. Launch the app
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`.

> The `data/` and `models/` folders are not stored in the repository (they are
> excluded in `.gitignore`); steps 4 and 5 recreate them. You only need to run
> them once.

## ⚙️ Switching Data Sources

By default the app reads the local CSV file produced by `generatedata.py`.
To use a real database instead:

1. Copy the template: `cp .env.example .env`
2. Set `CHURNLENS_DATA_SOURCE` to `sql` or `bigquery` and fill in the
   connection details.
3. Install the extra packages for that mode (they are optional and commented
   out in `requirements.txt`):
   - SQL: `pip install sqlalchemy psycopg2-binary` (MySQL: `pymysql`)
   - BigQuery: `pip install google-cloud-bigquery db-dtypes`
4. Restart the app.

The source table/query must return the same columns as the demo CSV.
`trainmodel.py` uses the same data layer, so the model can also be retrained
directly on the real source.

> ⚠️ Never commit your `.env` file — it may contain real credentials.
> `.gitignore` already excludes it.

## 📁 Project Structure

```
.
├── app.py                     # Streamlit UI (main application)
├── generatedata.py            # Synthetic demo data generator
├── trainmodel.py              # Model training & evaluation script
├── requirements.txt
├── .env.example               # Environment variable template
├── data_layer/                # Data source abstraction layer
│   ├── base.py                # Abstract contract (ChurnDataSource)
│   ├── csv_source.py          # CSV implementation
│   ├── sql_source.py          # Postgres/MySQL/SQLite implementation
│   ├── bigquery_source.py     # BigQuery implementation
│   ├── config.py              # Environment variable loading
│   └── factory.py             # Selects the right source
├── data/                      # (generated) saas_churn_data.csv
└── models/                    # (generated) churn_model.pkl, model_features.pkl,
                               #             categorical_options.pkl
```

## 🧰 Tech Stack

- **Streamlit** — web UI
- **scikit-learn** — churn prediction model and evaluation
- **SHAP** + **matplotlib** — model explainability
- **pandas / numpy** — data processing
- **python-dotenv** — `.env` configuration
- **SQLAlchemy**, **google-cloud-bigquery** — optional data source support
