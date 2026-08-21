"""
ChurnLens - Sentetik SaaS Müşteri Veri Üretici (v2)
-----------------------------------------------------
Önceki versiyonun temel sorunu: churn etiketi, özelliklerin doğrudan
ve neredeyse gürültüsüz bir fonksiyonuydu (özellikler de birbirinden
tamamen bağımsızdı). Bu yüzden model "gerçek bir örüntü" değil,
"formülün kendisini" öğreniyordu ve sonuçlar gerçekçi gelmiyordu.

Bu versiyonda:
  1. Özellikler birbirinden BAĞIMSIZ değil - tenure, kullanım,
     özellik benimseme, destek kayıtları ve ödeme gecikmesi
     arasında gerçekçi korelasyonlar var.
  2. Churn etiketi çok daha gürültülü (gerçek dünyada davranışın
     büyük kısmı gözlemlenemeyen faktörlerden etkilenir).
  3. Churn oranı gerçekçi bir seviyeye (~%14) çekildi; gerçek
     SaaS ürünlerinde çoğunlukla dengesiz bir dağılım görülür.
  4. Bazı sütunlarda gerçekçi eksik veri (missing values) var -
     gerçek analytics/CRM verisinde bu neredeyse her zaman olur.
  5. company_size ve signup_channel gibi iki kategorik değişken
     eklendi; bunlar hem gerçekçiliği artırıyor hem de fiyat/
     kullanım ile mantıklı şekilde ilişkilendirildi.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

N_CUSTOMERS = 5000  # Daha büyük ve daha gerçekçi bir portföy büyüklüğü

# ------------------------------------------------------------------
# 1. Temel kimlik ve firmografik alanlar
# ------------------------------------------------------------------
customer_ids = np.arange(10001, 10001 + N_CUSTOMERS)

company_size = rng.choice(
    ["Small", "Medium", "Enterprise"], size=N_CUSTOMERS, p=[0.55, 0.32, 0.13]
)
signup_channel = rng.choice(
    ["Organic", "Paid", "Referral", "Sales-Led"],
    size=N_CUSTOMERS,
    p=[0.35, 0.30, 0.20, 0.15],
)

# ------------------------------------------------------------------
# 2. Abonelik süresi (tenure) - company_size'a hafif bağlı:
#    Enterprise müşteriler ortalama biraz daha uzun soluklu olur.
# ------------------------------------------------------------------
tenure_base = rng.integers(1, 49, size=N_CUSTOMERS).astype(float)
tenure_bonus = np.where(company_size == "Enterprise", rng.integers(0, 12, N_CUSTOMERS), 0)
tenure_months = np.clip(tenure_base + tenure_bonus, 1, 60).astype(int)

# ------------------------------------------------------------------
# 3. Sözleşme tipi - Enterprise/Sales-Led kanalı yıllık sözleşmeye
#    daha yatkın; Organic/küçük şirketler aylığa daha yatkın.
# ------------------------------------------------------------------
annual_prob = 0.25
annual_prob = annual_prob + 0.30 * (company_size == "Enterprise") \
                          + 0.15 * (company_size == "Medium") \
                          + 0.20 * (signup_channel == "Sales-Led")
annual_prob = np.clip(annual_prob, 0.05, 0.9)
contract_type = np.array([
    rng.choice(["Annual", "Monthly"], p=[p, 1 - p]) for p in annual_prob
])

# ------------------------------------------------------------------
# 4. Aylık ücret - company_size ve sözleşme tipine bağlı, üstüne
#    gerçekçi gürültü eklendi.
# ------------------------------------------------------------------
size_price_base = np.select(
    [company_size == "Small", company_size == "Medium", company_size == "Enterprise"],
    [35.0, 70.0, 150.0],
)
contract_discount = np.where(contract_type == "Annual", 0.82, 1.0)  # yıllığa %18 indirim
monthly_charges = (
    size_price_base * contract_discount * rng.normal(1.0, 0.18, N_CUSTOMERS)
).clip(15, 400).round(2)

# ------------------------------------------------------------------
# 5. Kullanım davranışı - birbirine bağlı üç değişken:
#    usage_frequency yüksekse days_since_last_login düşük olma eğiliminde,
#    tenure arttıkça feature_adoption_rate ortalama olarak artıyor.
# ------------------------------------------------------------------
# Gizli bir "engagement" eğilimi (0-1) üretip, gözlemlenen değişkenleri
# bunun üstüne gürültü ekleyerek türetiyoruz -> gerçek korelasyon yapısı.
engagement_latent = rng.beta(2.2, 2.0, N_CUSTOMERS)

usage_frequency = np.clip(
    (engagement_latent * 27 + rng.normal(0, 4, N_CUSTOMERS)), 1, 30
).round().astype(int)

days_since_last_login = np.clip(
    ((1 - engagement_latent) * 28 + rng.normal(0, 5, N_CUSTOMERS)), 0, 30
).round().astype(int)

tenure_effect = np.clip(tenure_months / 48, 0, 1)
feature_adoption_rate = np.clip(
    (0.35 * tenure_effect + 0.55 * engagement_latent) * 100
    + rng.normal(0, 12, N_CUSTOMERS),
    5, 100
).round(1)

# ------------------------------------------------------------------
# 6. Destek kayıtları ve ödeme gecikmesi - hafif pozitif korelasyonlu
#    (sorun yaşayan müşteri hem çok ticket açar hem ödemede gecikir),
#    ayrıca engagement düşükse ticket sayısı artar.
# ------------------------------------------------------------------
friction_latent = rng.gamma(shape=1.4, scale=1.0, size=N_CUSTOMERS)
friction_latent = friction_latent + (1 - engagement_latent) * 0.8

support_tickets = rng.poisson(lam=np.clip(friction_latent, 0.2, None))
support_tickets = np.clip(support_tickets, 0, 15)

payment_delay_base = np.select(
    [friction_latent < 1, friction_latent < 2, friction_latent < 3],
    [0, 2, 5],
    default=10,
)
payment_delay_days = np.clip(
    payment_delay_base + rng.choice([0, 0, 0, 5, 15], size=N_CUSTOMERS, p=[0.55, 0.15, 0.1, 0.12, 0.08]),
    0, 60
)

# ------------------------------------------------------------------
# 7. NPS skoru (Net Promoter Score, -100..100) - engagement ve
#    destek deneyimiyle ilişkili, ~%6 oranında eksik (gerçek hayatta
#    anket doldurmayan müşteriler NaN kalır).
# ------------------------------------------------------------------
nps_score = np.clip(
    (engagement_latent - 0.5) * 140 - support_tickets * 4 + rng.normal(0, 20, N_CUSTOMERS),
    -100, 100
).round().astype(float)
missing_nps_mask = rng.random(N_CUSTOMERS) < 0.06
nps_score[missing_nps_mask] = np.nan

# feature_adoption_rate'te de gerçekçi bir eksik veri oranı (analytics
# takip kodu her müşteride tam çalışmayabilir)
missing_adoption_mask = rng.random(N_CUSTOMERS) < 0.03
feature_adoption_rate = feature_adoption_rate.astype(float)
feature_adoption_rate[missing_adoption_mask] = np.nan

# ------------------------------------------------------------------
# 8. Churn skoru - eskisinden çok daha gürültülü ve daha zayıf
#    sinyalli (gerçek dünyada davranışın büyük kısmı gözlemlenemeyen
#    faktörlerden etkilenir). NaN'ler skor hesaplamasında 0 kabul edilir.
# ------------------------------------------------------------------
nps_filled = np.nan_to_num(nps_score, nan=0.0)
adoption_filled = np.nan_to_num(feature_adoption_rate, nan=50.0)

churn_score = (
    0.028 * days_since_last_login
    + 0.09 * np.clip(support_tickets, 0, 6)
    + 0.02 * np.clip(payment_delay_days, 0, 30)
    + 0.35 * (contract_type == "Monthly")
    - 0.012 * adoption_filled
    - 0.015 * tenure_months
    - 0.006 * nps_filled
    + rng.normal(0, 0.9, N_CUSTOMERS)  # baskın gürültü terimi
)

# Sabit 0.5 eşiği yerine, hedef churn oranını (~%14) veren bir
# eşik (quantile) kullanıyoruz -> gerçekçi/dengesiz sınıf dağılımı.
target_churn_rate = 0.14
threshold = np.quantile(churn_score, 1 - target_churn_rate)
churn = (churn_score > threshold).astype(int)

# ------------------------------------------------------------------
# 9. DataFrame ve kayıt
# ------------------------------------------------------------------
df = pd.DataFrame(
    {
        "customer_id": customer_ids,
        "company_size": company_size,
        "signup_channel": signup_channel,
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "monthly_charges": monthly_charges,
        "days_since_last_login": days_since_last_login,
        "usage_frequency": usage_frequency,
        "feature_adoption_rate": feature_adoption_rate,
        "support_tickets": support_tickets,
        "payment_delay_days": payment_delay_days,
        "nps_score": nps_score,
        "churn": churn,
    }
)

df.to_csv("data/saas_churn_data.csv", index=False)

print("Dataset basariyla uretildi ve 'data/saas_churn_data.csv' dosyasina kaydedildi!")
print(f"Toplam Satir: {len(df)}")
print(f"Churn Dagilimi:\n{df['churn'].value_counts(normalize=True).round(3)}")
print(f"\nEksik Veri Oranlari:\n{df.isna().mean().round(3)}")
print(f"\nOzellik Korelasyonlari (sayisal):\n{df.select_dtypes('number').corr(numeric_only=True)['churn'].round(3)}")