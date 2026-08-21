"""
SQL Veri Kaynagi - Postgres, MySQL, SQL Server, SQLite... hepsi icin
TEK implementasyon. Hangisine baglandigimizi SQLAlchemy connection
string'i belirler, kod tarafinda fark yoktur.

Ornek connection string'ler:
  Postgres : postgresql+psycopg2://user:pass@host:5432/dbname
  MySQL    : mysql+pymysql://user:pass@host:3306/dbname
  SQLite   : sqlite:///local.db   (test/gelistirme icin)

Kurulum notu: bu modul sadece "sql" kaynagi secildiginde import edilir
(bkz. factory.py) -> CSV modunda calisirken SQLAlchemy kurulu olmasa
bile uygulama patlamaz.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .base import ChurnDataSource


class SQLDataSource(ChurnDataSource):
    name = "sql"

    def __init__(self, connection_string: str, query: str, engine: Engine | None = None):
        if not connection_string and engine is None:
            raise ValueError("SQLDataSource icin connection_string veya engine gerekli.")
        self.connection_string = connection_string
        self.query = query
        # pool_pre_ping: kopmus baglantilari otomatik tespit edip yeniler -
        # uzun sureli Streamlit proseslerinde onemli.
        self.engine = engine or create_engine(connection_string, pool_pre_ping=True)

    def _fetch_raw(self, include_label: bool) -> pd.DataFrame:
        query = self.query
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        if not include_label and "churn" in df.columns:
            df = df.drop(columns=["churn"])
        return df

    def health_check(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001
            return False