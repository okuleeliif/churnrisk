"""
ChurnLens - Model Egitim Scripti (v2)
----------------------------------------
Onceki versiyondaki eksikler ve bu versiyondaki cozumler:

  1. Sabit hiperparametreler yerine RandomizedSearchCV ile arama.
  2. Sinif dengesizligi (~%14 churn) icin class_weight="balanced".
  3. Tek train/test split yerine, arama sirasinda StratifiedKFold ile
     capraz dogrulama; disaridan tek bir "gormedigi" test seti ile
     nihai, tarafsiz degerlendirme.
  4. Sadece ROC-AUC degil, dengesiz siniflarda daha anlamli olan
     PR-AUC (average precision) ve confusion matrix de raporlaniyor.
  5. HistGradientBoostingClassifier eksik degerleri (NaN) NATIF olarak
     destekler -> ayri bir imputation adimina gerek yok, bu da
     app.py tarafinda da ayni mantigin kullanilmasini saglar.
  6. Kategorik degiskenler (contract_type, company_size,
     signup_channel) one-hot encode ediliyor; kolon sirasi/isimleri
     model_features.pkl icinde saklaniyor ki app.py ile birebir
     ayni sema kullanilsin.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

from data_layer import get_data_source

RANDOM_SEED = 42

# ------------------------------------------------------------------
# 1. Veri setini yukleme - artik dogrudan CSV degil, data_layer
#    uzerinden okunuyor. CHURNLENS_DATA_SOURCE ortam degiskeni "sql"
#    veya "bigquery" olarak ayarlanirsa, model DOGRUDAN GERCEK
#    VERITABANINDAN egitilir; bu dosyada hicbir degisiklik gerekmez.
# ------------------------------------------------------------------
data_source = get_data_source()
print(f"Veri kaynagi: {data_source.name}")
df = data_source.fetch_customers(include_label=True)

CATEGORICAL_COLS = ["contract_type", "company_size", "signup_channel"]
df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

X = df_encoded.drop(columns=["customer_id", "churn"])
y = df_encoded["churn"]
model_features = X.columns.tolist()

# ------------------------------------------------------------------
# 2. Egitim / Test ayrimi (stratify ile churn oranini her iki sette
#    de koruyoruz - dengesiz siniflar icin onemli)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)

# ------------------------------------------------------------------
# 3. Hiperparametre aramasi (RandomizedSearchCV + StratifiedKFold)
#    ROC-AUC yerine dengesiz siniflarda daha guvenilir olan
#    "average_precision" (PR-AUC) skoruna gore optimize ediyoruz.
# ------------------------------------------------------------------
param_distributions = {
    "learning_rate": [0.02, 0.05, 0.08, 0.1],
    "max_depth": [3, 4, 5, 6, None],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [10, 20, 40, 60],
    "l2_regularization": [0.0, 0.1, 0.5, 1.0],
    "max_iter": [150, 250, 350],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

base_model = HistGradientBoostingClassifier(
    class_weight="balanced",  # ~%14/%86 dengesizligini modele bildiriyoruz
    random_state=RANDOM_SEED,
    early_stopping=True,
    validation_fraction=0.15,
)

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=25,
    scoring="average_precision",
    cv=cv,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    refit=True,
)

print("Hiperparametre aramasi basliyor (5-fold CV, 25 kombinasyon)...")
search.fit(X_train, y_train)

model = search.best_estimator_
print(f"\nEn iyi parametreler:\n{search.best_params_}")
print(f"CV PR-AUC (en iyi): {search.best_score_:.4f}")

# ------------------------------------------------------------------
# 4. Nihai, gorulmemis test seti uzerinde tarafsiz degerlendirme
# ------------------------------------------------------------------
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Model Performans Raporu (Test Seti) ---")
print(classification_report(y_test, y_pred, target_names=["Kalici", "Churn"]))
print(f"ROC-AUC Skoru : {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"PR-AUC Skoru  : {average_precision_score(y_test, y_pred_proba):.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nKarisiklik Matrisi (satir=gercek, sutun=tahmin) [Kalici, Churn]:")
print(cm)

# Basit bir "naive baseline" karsilastirmasi: her seyi coguncluk
# sinifi (Kalici) olarak tahmin etsek ne olurdu? Modelin gercekten
# bir sey ogrenip ogrenmedigini gormek icin faydali bir referans.
naive_accuracy = (y_test == 0).mean()
print(f"\nReferans: Hep 'Kalici' tahmin etseydik dogruluk = {naive_accuracy:.4f}")
print(f"Modelin gercek dogrulugu               = {(y_pred == y_test).mean():.4f}")

# ------------------------------------------------------------------
# 5. Modeli ve semayi kaydetme
#    Kategorik sutunlarin gercek dunyada gorulen degerlerini de
#    kaydediyoruz ki app.py ayni secenekleri (dropdown) UI'da
#    dogru sekilde uretebilsin.
# ------------------------------------------------------------------
categorical_options = {col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL_COLS}

os.makedirs("models", exist_ok=True)  # Repo yeni indirildiyse models/ klasoru olmayabilir
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(model_features, "models/model_features.pkl")
joblib.dump(categorical_options, "models/categorical_options.pkl")

print("\nModel, ozellik listesi ve kategorik secenekler 'models/' klasorune kaydedildi!")