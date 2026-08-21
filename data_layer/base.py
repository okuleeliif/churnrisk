"""
ChurnLens Veri Katmani - Temel Soyutlama (Abstract Base)
----------------------------------------------------------
Bu modul, uygulamanin "musteri verisini nereden okudugunu"
bilmesine GEREK BIRAKMAYAN bir sozlesme (interface) tanimlar.

Neden bu katman var?
  app.py ve train_model.py su an dogrudan `pd.read_csv(...)` cagirsaydi,
  yarin gercek bir Postgres/BigQuery kaynagina gecmek istediginde HER
  IKI dosyayi da degistirmen gerekirdi. Bunun yerine tum kod tabani
  sadece `ChurnDataSource.fetch_customers()` sozlesmesini bilir; hangi
  somut sinifin (CSV, SQL, BigQuery...) calisacagi tamamen bir ortam
  degiskeniyle (bkz. config.py / factory.py) belirlenir.

  Bu, "Strategy / Adapter" tasarim deseni olarak bilinir ve bircok
  gercek veri platformunda (orkestrasyon katmani) ayni sekilde kullanilir.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger("churnlens.data_layer")

# ------------------------------------------------------------------
# Beklenen sema. "required" olmayan alanlar gercek dunyada henuz
# toplanmiyor olabilir (ornegin yeni entegre olan bir sirket NPS
# anketi yapmiyor olabilir) -> eksikse otomatik NaN ile doldurulur,
# uygulama CALISMAYA DEVAM EDER, sadece bir uyari loglanir.
# ------------------------------------------------------------------
REQUIRED_COLUMNS = {
    "customer_id": "int64",
    "tenure_months": "int64",
    "contract_type": "object",
    "monthly_charges": "float64",
    "days_since_last_login": "int64",
    "usage_frequency": "int64",
    "support_tickets": "int64",
    "payment_delay_days": "int64",
}

OPTIONAL_COLUMNS = {
    "company_size": "object",
    "signup_channel": "object",
    "feature_adoption_rate": "float64",
    "nps_score": "float64",
    "churn": "int64",  # sadece egitim modunda anlamli; canli skorlamada olmayabilir
}

ALL_KNOWN_COLUMNS = {**REQUIRED_COLUMNS, **OPTIONAL_COLUMNS}


class SchemaValidationError(Exception):
    """Zorunlu bir kolon tamamen eksikse (kaynak yanlis yapilandirilmis demektir)."""


class ChurnDataSource(ABC):
    """Tum veri kaynaklarinin uymasi gereken sozlesme."""

    #: Alt siniflar bunu override eder (ornek: "csv", "postgres", "bigquery")
    name: str = "base"

    @abstractmethod
    def _fetch_raw(self, include_label: bool) -> pd.DataFrame:
        """Somut kaynaktan ham DataFrame'i ceker. Alt siniflar bunu implemente eder."""
        raise NotImplementedError

    def fetch_customers(self, include_label: bool = True) -> pd.DataFrame:
        """
        Disariya acik TEK giris noktasi. Ham veriyi ceker, semayi
        dogrular/tamamlar ve tutarli bir DataFrame doner.
        """
        df = self._fetch_raw(include_label=include_label)
        df = self._validate_and_align_schema(df, include_label=include_label)
        logger.info(
            "[%s] %d satir okundu (include_label=%s).",
            self.name, len(df), include_label,
        )
        return df

    def _validate_and_align_schema(self, df: pd.DataFrame, include_label: bool) -> pd.DataFrame:
        required = dict(REQUIRED_COLUMNS)
        optional = dict(OPTIONAL_COLUMNS)
        if not include_label:
            optional.pop("churn", None)

        missing_required = [c for c in required if c not in df.columns]
        if missing_required:
            raise SchemaValidationError(
                f"[{self.name}] Zorunlu kolon(lar) eksik: {missing_required}. "
                "Veri kaynagi yapilandirmasini (sorgu/tablo) kontrol edin."
            )

        missing_optional = [c for c in optional if c not in df.columns]
        for col in missing_optional:
            df[col] = pd.NA
            logger.warning(
                "[%s] Opsiyonel kolon '%s' kaynakta bulunamadi, NaN ile dolduruldu. "
                "Model bunu native olarak eksik veri gibi ele alacak.",
                self.name, col,
            )

        # Fazladan kolonlar varsa (gercek DB'lerde sik gorulur) sadece bilgi ver, at ma.
        extra_cols = [c for c in df.columns if c not in ALL_KNOWN_COLUMNS]
        if extra_cols:
            logger.info("[%s] Bilinmeyen ek kolonlar goz ardi edildi: %s", self.name, extra_cols)

        ordered_cols = [c for c in {**required, **optional} if c in df.columns]
        return df[ordered_cols].reset_index(drop=True)

    def health_check(self) -> bool:
        """Baglanti/erisim saglikli mi? UI'da yesil/kirmizi gostermek icin kullanilir."""
        try:
            self._fetch_raw(include_label=False)
            return True
        except Exception as exc:  # noqa: BLE001 - kasitli genis yakalama, sadece durum bildirimi icin
            logger.error("[%s] Health check basarisiz: %s", self.name, exc)
            return False