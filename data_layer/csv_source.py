"""CSV Veri Kaynagi - demo/gelistirme modu (varsayilan)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import ChurnDataSource


class CSVDataSource(ChurnDataSource):
    name = "csv"

    def __init__(self, path: str = "data/saas_churn_data.csv"):
        self.path = Path(path)

    def _fetch_raw(self, include_label: bool) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"CSV veri dosyasi bulunamadi: {self.path}. "
                "Once 'python generate_data.py' calistirilmali."
            )
        df = pd.read_csv(self.path)
        if not include_label and "churn" in df.columns:
            df = df.drop(columns=["churn"])
        return df