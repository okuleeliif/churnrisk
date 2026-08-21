"""
Factory - "orkestrasyon" katmaninin girisi.

app.py ve train_model.py TEK SATIR ile veri kaynagini alir:

    from data_layer import get_data_source
    df = get_data_source().fetch_customers()

Hangi somut kaynagin donecegi tamamen `CHURNLENS_DATA_SOURCE` ortam
degiskenine baglidir. Kodda hicbir if/else degisikligi gerekmez.
"""

from __future__ import annotations

from .base import ChurnDataSource
from .config import DataSourceConfig
from .csv_source import CSVDataSource


def get_data_source(config: DataSourceConfig | None = None) -> ChurnDataSource:
    config = config or DataSourceConfig()

    if config.source_type == "csv":
        return CSVDataSource(path=config.csv_path)

    if config.source_type in ("sql", "postgres", "postgresql", "mysql"):
        if not config.sql_connection_string:
            raise ValueError(
                "CHURNLENS_DATA_SOURCE='sql' secildi ama CHURNLENS_SQL_CONN "
                "(baglanti string'i) tanimli degil. .env.example dosyasina bakin."
            )
        # SQLAlchemy sadece bu dalda gerekli -> lazy import
        from .sql_source import SQLDataSource
        return SQLDataSource(
            connection_string=config.sql_connection_string,
            query=config.effective_sql_query(),
        )

    if config.source_type == "bigquery":
        # google-cloud-bigquery sadece bu dalda gerekli -> lazy import
        from .bigquery_source import BigQueryDataSource
        return BigQueryDataSource(project=config.bq_project, query=config.bq_query)

    raise ValueError(
        f"Bilinmeyen CHURNLENS_DATA_SOURCE degeri: '{config.source_type}'. "
        "Gecerli degerler: 'csv', 'sql', 'bigquery'."
    )