"""
BigQuery Veri Kaynagi.

`google-cloud-bigquery` paketi sadece bu kaynak SECILDIGINDE import
edilir (asagida fonksiyon icinde lazy import). Boylece CSV/SQL modunda
calisan kullanicilarin bu agir/opsiyonel bagimliligi kurmasi gerekmez.

Kimlik dogrulama: standart Google Application Default Credentials (ADC)
kullanilir - ortamda `GOOGLE_APPLICATION_CREDENTIALS` env degiskeni
service account json dosyasini gostermelidir.
"""

from __future__ import annotations

import pandas as pd

from .base import ChurnDataSource


class BigQueryDataSource(ChurnDataSource):
    name = "bigquery"

    def __init__(self, project: str, query: str):
        if not project or not query:
            raise ValueError("BigQueryDataSource icin 'project' ve 'query' zorunludur.")
        self.project = project
        self.query = query
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from google.cloud import bigquery  # lazy import
            except ImportError as exc:
                raise ImportError(
                    "BigQuery kaynagi icin 'google-cloud-bigquery' paketi gerekli: "
                    "pip install google-cloud-bigquery"
                ) from exc
            self._client = bigquery.Client(project=self.project)
        return self._client

    def _fetch_raw(self, include_label: bool) -> pd.DataFrame:
        df = self.client.query(self.query).to_dataframe()
        if not include_label and "churn" in df.columns:
            df = df.drop(columns=["churn"])
        return df

    def health_check(self) -> bool:
        try:
            list(self.client.query("SELECT 1").result())
            return True
        except Exception:  # noqa: BLE001
            return False