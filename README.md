# 📊 ChurnLens — SaaS Customer Churn Prediction & Decision Support Platform

ChurnLens is a Streamlit application that analyzes a SaaS company's customer
portfolio, predicts which customers are at risk of churning using machine
learning, and provides actionable, **What-If simulations** ("what happens to
the risk if this customer's support tickets go down?") to support retention
decisions.

## ✨ Features

- **Portfolio-wide risk scoring** — analyzes the entire customer base,
  including MRR (monthly recurring revenue) at risk.
- **Automatic segmentation** — buckets at-risk customers into 4 operational
  clusters by root cause (onboarding, engagement drop, support crisis,
  contract/payment risk).
- **What-If simulation** — instantly see how risk changes when you adjust a
  customer's attributes (contract type, support tickets, usage frequency,
  etc.).
- **SHAP explainability** — visualizes exactly why the model made a given
  prediction, broken down by feature contribution.
- **Automated intervention report** — generates a downloadable "rescue plan"
  report for the selected customer.
- **Pluggable data layer** — demo with CSV today, connect to a real
  Postgres/MySQL/BigQuery source tomorrow with zero code changes (see
  [Switching Data Sources](#️-switching-data-sources)).

## 🏗️ Architecture

```
generate_data.py  →  data/saas_churn_data.csv  →  data_layer/  →  train_model.py  →  models/*.pkl
                                                        │
                                                        ▼
                                                     app.py (Streamlit UI)
```

The `data_layer/` package fully decouples the data source (CSV / SQL /
BigQuery) from the rest of the application — which source is used is
determined entirely by an environment variable, with no code changes
required.

## 🚀 Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) SQL mode dependencies are already included in
# requirements.txt (sqlalchemy / psycopg2-binary)

# 3. Generate the demo dataset
python generate_data.py

# 4. Train the model
python train_model.py

# 5. Launch the app
streamlit run app.py
```

## ⚙️ Switching Data Sources

By default, the app reads the local CSV file produced by `generate_data.py`.
To connect to a real database instead:

1. Copy `.env.example` to `.env`.
2. Fill in the desired mode (`sql` or `bigquery`) and connection details.
3. Restart the app — it connects to the new source with zero code changes.

See `.env.example` for detailed examples and the full list of environment
variables.

> ⚠️ Never commit your `.env` file — it may contain real credentials /
> connection strings. `.gitignore` already excludes it by default.

## 📁 Project Structure

```
.
├── app.py                    # Streamlit UI (main application)
├── generate_data.py           # Synthetic demo data generator
├── train_model.py             # Model training script
├── requirements.txt
├── .env.example                # Environment variable template
├── data/
│   └── saas_churn_data.csv     # Generated demo data (output of generate_data.py)
├── models/
│   ├── churn_model.pkl          # Trained model
│   ├── model_features.pkl       # Model feature list
│   └── categorical_options.pkl  # Categorical field options
└── data_layer/                 # Data source abstraction layer
    ├── base.py                   # Abstract contract (ChurnDataSource)
    ├── csv_source.py              # CSV implementation
    ├── sql_source.py               # Postgres/MySQL/SQLite implementation
    ├── bigquery_source.py          # BigQuery implementation
    ├── config.py                    # Environment variable loading
    └── factory.py                    # Factory that selects the right source
```

## 🧰 Tech Stack

- **Streamlit** — web UI
- **scikit-learn** (`HistGradientBoostingClassifier`) — churn prediction model
- **SHAP** — model explainability
- **pandas / numpy** — data processing
- **SQLAlchemy** — SQL data source support (optional)
- **google-cloud-bigquery** — BigQuery data source support (optional)

## 📝 Note

The files in `data/` and `models/` can be regenerated at any time by running
`generate_data.py` and `train_model.py`. If you'd like to keep the repo
lightweight, you can exclude them via `.gitignore` — the relevant lines are
already there, commented out, ready to enable.

## 📄 License

Add your preferred license here (e.g. MIT, Apache 2.0).
