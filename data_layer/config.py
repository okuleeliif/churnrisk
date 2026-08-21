"""
Veri kaynagi yapilandirmasi - tamamen ortam degiskenlerinden (env vars)
okunur. Kod DEGISMEZ, sadece calisma ortaminda .env / deployment
degiskenleri degisir.

Kullanim:
  CSV (varsayilan, demo)  -> hicbir sey ayarlamana gerek yok
  Postgres/MySQL          -> CHURNLENS_DATA_SOURCE=sql
                              CHURNLENS_SQL_CONN=postgresql+psycopg2://...
                              CHURNLENS_SQL_QUERY="SELECT * FROM customers_churn_view"
  BigQuery                -> CHURNLENS_DATA_SOURCE=bigquery
                              CHURNLENS_BQ_PROJECT=my-gcp-project
                              CHURNLENS_BQ_QUERY="SELECT * FROM `proj.dataset.churn_view`"

Ayrintili ornekler icin .env.example dosyasina bakin.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _load_dotenv_if_present() -> None:
    """python-dotenv kuruluysa ve .env varsa otomatik yukler (opsiyonel)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv kurulu degilse sessizce gec - env vars zaten set edilmis olabilir


_load_dotenv_if_present()


@dataclass
class DataSourceConfig:
    source_type: str = field(default_factory=lambda: os.getenv("CHURNLENS_DATA_SOURCE", "csv").lower())

    # --- CSV modu ---
    csv_path: str = field(default_factory=lambda: os.getenv("CHURNLENS_CSV_PATH", "data/saas_churn_data.csv"))

    # --- SQL modu (Postgres/MySQL/SQLite...) ---
    sql_connection_string: str = field(default_factory=lambda: os.getenv("CHURNLENS_SQL_CONN", ""))
    sql_table: str = field(default_factory=lambda: os.getenv("CHURNLENS_SQL_TABLE", "customers_churn"))
    sql_query: str = field(default_factory=lambda: os.getenv("CHURNLENS_SQL_QUERY", ""))

    # --- BigQuery modu ---
    bq_project: str = field(default_factory=lambda: os.getenv("CHURNLENS_BQ_PROJECT", ""))
    bq_query: str = field(default_factory=lambda: os.getenv("CHURNLENS_BQ_QUERY", ""))

    def effective_sql_query(self) -> str:
        return self.sql_query or f"SELECT * FROM {self.sql_table}"