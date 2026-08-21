"""
ChurnLens Veri Katmani (Data Layer)
--------------------------------------
Dis dunyaya (app.py, train_model.py) sadece bunlari acar:

    get_data_source()      -> aktif ortam degiskenlerine gore
                               CSVDataSource / SQLDataSource /
                               BigQueryDataSource orneklerinden birini doner
    ChurnDataSource         -> tum kaynaklarin uydugu soyut sozlesme
    SchemaValidationError   -> zorunlu kolon eksikse firlatilir
    DataSourceConfig        -> mevcut yapilandirmayi programatik okumak icin
"""

from .base import ChurnDataSource, SchemaValidationError
from .config import DataSourceConfig
from .factory import get_data_source

__all__ = [
    "get_data_source",
    "ChurnDataSource",
    "SchemaValidationError",
    "DataSourceConfig",
]