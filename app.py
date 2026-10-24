# ==============================================================================
# UYGULAMAYI CALISTIRMAK ICIN (Terminal'e sirayla yapistir):
#
#   cd ~/Desktop/churnlens
#   source venv/bin/activate
#   python -m streamlit run app.py
#
# Durdurmak icin: Ctrl + C
# ==============================================================================

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from data_layer import get_data_source

# Sayfa Yapılandırması
st.set_page_config(
    page_title="ChurnLens - SaaS Retention & What-If Studio",
    page_icon="📊",
    layout="wide",
)

CATEGORICAL_COLS = ["contract_type", "company_size", "signup_channel"]


# ==============================================================================
# 1. Model, Özellikler ve Veriyi Yükleme & Önbellekleme
#    (@st.cache_resource -> tek proses omru boyunca bir kez yuklenir)
#    Veri artik dogrudan CSV degil, data_layer uzerinden okunuyor.
#    CHURNLENS_DATA_SOURCE ortam degiskeni degistirilerek uygulama
#    KOD DEGISTIRILMEDEN gercek bir Postgres/BigQuery kaynagina
#    baglanabilir (bkz. .env.example).
# ==============================================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/churn_model.pkl")
    features = joblib.load("models/model_features.pkl")
    categorical_options = joblib.load("models/categorical_options.pkl")
    data_source = get_data_source()
    df = data_source.fetch_customers(include_label=True)
    explainer = shap.TreeExplainer(model)
    return model, features, categorical_options, df, explainer, data_source


model, model_features, categorical_options, df_raw, explainer, data_source = load_artifacts()

with st.sidebar:
    st.markdown("### ⚙️ Veri Kaynağı")
    healthy = data_source.health_check()
    status_icon = "🟢" if healthy else "🔴"
    st.markdown(f"{status_icon} **Aktif Kaynak:** `{data_source.name}`")
    if data_source.name == "csv":
        st.caption("Demo modu. Gerçek bir veritabanına bağlamak için `CHURNLENS_DATA_SOURCE` ortam değişkenini `sql` veya `bigquery` yapın (bkz. `.env.example`).")
    elif not healthy:
        st.error("Bağlantı sağlıksız görünüyor — yapılandırmayı ve kimlik bilgilerini kontrol edin.")


# ==============================================================================
# 2. Tüm Portföyü Skorlama - AYRI bir cache katmanı.
#    Buyuk veri setlerinde (100k-1M+ satir) bu adim maliyetli olabilir;
#    @st.cache_data sayesinde veri/model degismedigi surece tekrar
#    hesaplanmaz (session'lar arasi da paylasilir, sadece ilk kullanici
#    bekler). Not: HistGradientBoostingClassifier eksik degerleri (NaN)
#    NATIF destekler, bu yuzden ayri bir imputation adimina gerek yok.
# ==============================================================================
@st.cache_data
def score_portfolio(df_raw: pd.DataFrame, _model, features: list[str]) -> pd.DataFrame:
    df_encoded = pd.get_dummies(df_raw, columns=CATEGORICAL_COLS, drop_first=True)
    # Egitimde gorulmeyen/gorulmus ama bu batch'te olusmayan dummy kolonlarini hizala
    for col in features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    X_all = df_encoded[features]

    scored = df_raw.copy()
    scored["churn_probability"] = _model.predict_proba(X_all)[:, 1]
    scored["predicted_churn"] = _model.predict(X_all)
    return scored


df_raw = score_portfolio(df_raw, model, model_features)

# Başlık
st.title("📊 ChurnLens: SaaS Müşteri Kayıp & Karar Destek Platformu")
st.caption(
    "Portföy risklerini izleyin, segment bazlı aksiyon alın ve tekil What-If simülasyonları gerçekleştirin."
)
st.caption(
    "ℹ️ Model, gerçekçi şekilde eksik veri (`feature_adoption_rate`, `nps_score`) ve "
    "dengesiz sınıf dağılımı (~%14 churn) içerecek şekilde eğitildi."
)
st.divider()

# ==============================================================================
# BÖLÜM 1: ŞİRKET GENELİ PORTFÖY ÖZETİ
# ==============================================================================
total_customers = len(df_raw)
predicted_churners = int(df_raw["predicted_churn"].sum())
churn_rate = (predicted_churners / total_customers) * 100
at_risk_mrr = df_raw[df_raw["predicted_churn"] == 1]["monthly_charges"].sum()
total_mrr = df_raw["monthly_charges"].sum()

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Toplam Müşteri", f"{total_customers:,}")
col_m2.metric(
    "Risk Altındaki Müşteri",
    f"{predicted_churners:,}",
    f"%{churn_rate:.1f} Kayıp Oranı",
    delta_color="inverse",
)
col_m3.metric(
    "Tehlikedeki Aylık Ciro ($MRR)",
    f"${at_risk_mrr:,.0f}",
    f"%{(at_risk_mrr / total_mrr) * 100:.1f} Pay",
    delta_color="inverse",
)
col_m4.metric("Toplam Aylık Ciro ($MRR)", f"${total_mrr:,.0f}")

st.divider()

# ==============================================================================
# BÖLÜM 2: 4'LÜ TOPLU RİSK SEGMENTASYONU (MAKRO MÜDAHALE)
# ==============================================================================
st.subheader("🎯 1. Toplu Risk Segmentasyonu (Makro Müdahale)")
st.caption(
    "Risk altındaki müşteriler kök nedenlerine göre 4 operasyonel kümeye ayrılmıştır:"
)

seg_onboarding = df_raw[
    (df_raw["predicted_churn"] == 1)
    & (df_raw["tenure_months"] < 3)
    & (df_raw["feature_adoption_rate"].fillna(100) < 40)
]
seg_engagement = df_raw[
    (df_raw["predicted_churn"] == 1)
    & (df_raw["days_since_last_login"] > 10)
    & (~df_raw.index.isin(seg_onboarding.index))
]
seg_support = df_raw[
    (df_raw["predicted_churn"] == 1)
    & (df_raw["support_tickets"] >= 3)
    & (~df_raw.index.isin(seg_onboarding.index))
    & (~df_raw.index.isin(seg_engagement.index))
]
seg_commercial = df_raw[
    (df_raw["predicted_churn"] == 1)
    & (
        (df_raw["payment_delay_days"] > 0)
        | (df_raw["contract_type"] == "Monthly")
    )
    & (~df_raw.index.isin(seg_onboarding.index))
    & (~df_raw.index.isin(seg_engagement.index))
    & (~df_raw.index.isin(seg_support.index))
]

c_s1, c_s2, c_s3, c_s4 = st.columns(4)
c_s1.metric(
    "1. Onboarding Kırılganları",
    f"{len(seg_onboarding)} Müşteri",
    f"${seg_onboarding['monthly_charges'].sum():,.0f} MRR",
)
c_s2.metric(
    "2. Kullanımı Düşenler",
    f"{len(seg_engagement)} Müşteri",
    f"${seg_engagement['monthly_charges'].sum():,.0f} MRR",
)
c_s3.metric(
    "3. Destek Krizi Yaşayanlar",
    f"{len(seg_support)} Müşteri",
    f"${seg_support['monthly_charges'].sum():,.0f} MRR",
)
c_s4.metric(
    "4. Sözleşme & Ödeme Riski",
    f"{len(seg_commercial)} Müşteri",
    f"${seg_commercial['monthly_charges'].sum():,.0f} MRR",
)

with st.expander("📂 Segment Listelerini İncele ve İndir"):
    seg_choice = st.selectbox(
        "Görüntülenecek Segmenti Seçin:",
        [
            "1. Onboarding Kırılganları (CSM Eğitimi)",
            "2. Kullanımı Düşenler (Pazarlama Re-activation)",
            "3. Destek Krizi (Teknik Öncelik)",
            "4. Sözleşme & Ödeme Riski (Satış Kampanyası)",
        ],
    )
    mapping = {
        "1. Onboarding Kırılganları (CSM Eğitimi)": seg_onboarding,
        "2. Kullanımı Düşenler (Pazarlama Re-activation)": seg_engagement,
        "3. Destek Krizi (Teknik Öncelik)": seg_support,
        "4. Sözleşme & Ödeme Riski (Satış Kampanyası)": seg_commercial,
    }
    st.dataframe(mapping[seg_choice], width="stretch", hide_index=True)
    st.download_button(
        "⬇️ Bu Segmenti CSV Olarak İndir",
        data=mapping[seg_choice].to_csv(index=False).encode("utf-8"),
        file_name="churnlens_segment.csv",
        mime="text/csv",
    )

st.divider()

# ==============================================================================
# BÖLÜM 3: MÜŞTERİ ARAMA, SIRALAMA VE SEÇİM
# ==============================================================================
st.subheader("🔍 2. Müşteri İnceleme & Arama Merkezi")

# Arama ve Sıralama Filtre Çubuğu
col_search, col_sort = st.columns([2, 1])
with col_search:
    search_query = st.text_input(
        "🔎 Müşteri ID / No ile Ara:",
        placeholder="Örn: 1042 veya 10",
        help="Aramak istediğiniz müşteri numarasını girin.",
    )
with col_sort:
    sort_order = st.selectbox(
        "📊 Risk Sıralaması:",
        options=[
            "En Yüksek Riskten En Düşüğe (Azalan)",
            "En Düşük Riskten En Yükseğe (Artan)",
            "Aylık Gelire Göre (Yüksekten Düşüğe)",
        ],
    )

# Veriyi sıralama seçeneğine göre düzenleme
if sort_order == "En Yüksek Riskten En Düşüğe (Azalan)":
    df_filtered = df_raw.sort_values(by="churn_probability", ascending=False)
elif sort_order == "En Düşük Riskten En Yükseğe (Artan)":
    df_filtered = df_raw.sort_values(by="churn_probability", ascending=True)
else:
    df_filtered = df_raw.sort_values(by="monthly_charges", ascending=False)

# Arama filtresini uygulama
if search_query.strip():
    df_filtered = df_filtered[
        df_filtered["customer_id"].astype(str).str.contains(search_query.strip())
    ]

if len(df_filtered) == 0:
    st.warning("⚠️ Aradığınız kriterlere uygun müşteri bulunamadı.")
    st.stop()

customer_options = df_filtered["customer_id"].tolist()
_lookup = df_filtered.set_index("customer_id")

selected_customer_id = st.selectbox(
    f"Müşteri Seçin ({len(customer_options)} müşteri listeleniyor):",
    options=customer_options,
    format_func=lambda cid: f"Müşteri #{cid} | Orijinal Risk: %{_lookup.at[cid, 'churn_probability'] * 100:.1f} | Aylık Gelir: ${_lookup.at[cid, 'monthly_charges']:.2f}",
)

cust_row = df_raw[df_raw["customer_id"] == selected_customer_id].iloc[0]
original_prob = float(cust_row["churn_probability"])


def safe_int(value, default):
    """NaN gelebilecek alanlar icin guvenli int donusumu (eksik veri senaryosu)."""
    return default if pd.isna(value) else int(value)


def safe_float(value, default):
    return default if pd.isna(value) else float(value)


def reset_single_val(key_name, original_value):
    st.session_state[key_name] = original_value


# Eksik veri olabilecek alanlar icin populasyon medyanini varsayilan yapiyoruz
median_adoption = int(round(df_raw["feature_adoption_rate"].median()))
median_nps = int(round(df_raw["nps_score"].median()))

if (
    "active_customer_id" not in st.session_state
    or st.session_state["active_customer_id"] != selected_customer_id
):
    st.session_state["active_customer_id"] = selected_customer_id
    st.session_state["val_tenure_months"] = int(cust_row["tenure_months"])
    st.session_state["val_contract_type"] = str(cust_row["contract_type"])
    st.session_state["val_company_size"] = str(cust_row["company_size"])
    st.session_state["val_signup_channel"] = str(cust_row["signup_channel"])
    st.session_state["val_monthly_charges"] = float(cust_row["monthly_charges"])
    st.session_state["val_days_since_last_login"] = int(cust_row["days_since_last_login"])
    st.session_state["val_usage_frequency"] = int(cust_row["usage_frequency"])
    st.session_state["val_feature_adoption_rate"] = safe_int(
        cust_row["feature_adoption_rate"], median_adoption
    )
    st.session_state["val_support_tickets"] = int(cust_row["support_tickets"])
    st.session_state["val_payment_delay_days"] = int(cust_row["payment_delay_days"])
    st.session_state["val_nps_score"] = safe_int(cust_row["nps_score"], median_nps)

adoption_display = (
    "Veri Yok" if pd.isna(cust_row["feature_adoption_rate"]) else f"%{cust_row['feature_adoption_rate']:.0f}"
)
nps_display = "Veri Yok" if pd.isna(cust_row["nps_score"]) else f"{cust_row['nps_score']:.0f}"

summary_table = pd.DataFrame(
    [
        {
            "Müşteri ID": int(cust_row["customer_id"]),
            "Orijinal Risk": f"%{original_prob * 100:.1f}",
            "Orijinal Teşhis": "🚨 Terk Edebilir" if original_prob >= 0.5 else "✅ Güvenli",
            "Firma Büyüklüğü": cust_row["company_size"],
            "Kayıt Kanalı": cust_row["signup_channel"],
            "Abonelik Süresi": f"{cust_row['tenure_months']} Ay",
            "Sözleşme": cust_row["contract_type"],
            "Aylık Ücret": f"${cust_row['monthly_charges']:.2f}",
            "Son Giris": f"{cust_row['days_since_last_login']} Gün Önce",
            "Aktif Gün": f"{cust_row['usage_frequency']} Gün/Ay",
            "Özellik Kullanımı": adoption_display,
            "NPS": nps_display,
            "Destek Kaydı": f"{cust_row['support_tickets']} Adet",
            "Fatura Gecikmesi": f"{cust_row['payment_delay_days']} Gün",
        }
    ]
)
st.dataframe(summary_table, width="stretch", hide_index=True)

st.markdown("---")

# ==============================================================================
# BÖLÜM 4: SOLDA WHAT-IF BARLARI, SAĞDA GÜNCEL SONUÇ & SHAP
# ==============================================================================
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("🧪 3. What-If Simülasyon Barları")
    st.caption(
        "Değerleri kaydırın. Her alanın yanındaki **↺** butonuyla orijinal değerine dönebilirsiniz."
    )

    def slider_row(label, min_v, max_v, key, reset_val, help_text=None):
        c1, c2 = st.columns([5, 1])
        with c1:
            val = st.slider(label, min_v, max_v, key=key, help=help_text)
        with c2:
            st.write("")
            st.write("")
            st.button("↺", key=f"btn_reset_{key}", on_click=reset_single_val, args=(key, reset_val), help="Sıfırla")
        return val

    def select_row(label, options, key, reset_val):
        c1, c2 = st.columns([5, 1])
        with c1:
            val = st.selectbox(label, options, key=key)
        with c2:
            st.write("")
            st.write("")
            st.button("↺", key=f"btn_reset_{key}", on_click=reset_single_val, args=(key, reset_val), help="Sıfırla")
        return val

    def number_row(label, min_v, max_v, key, reset_val):
        c1, c2 = st.columns([5, 1])
        with c1:
            val = st.number_input(label, min_value=min_v, max_value=max_v, key=key)
        with c2:
            st.write("")
            st.write("")
            st.button("↺", key=f"btn_reset_{key}", on_click=reset_single_val, args=(key, reset_val), help="Sıfırla")
        return val

    sim_tenure = slider_row(
        f"Abonelik Süresi (Ay)  *(Orijinal: {cust_row['tenure_months']})*",
        1, 72, "val_tenure_months", int(cust_row["tenure_months"]),
    )
    sim_contract = select_row(
        f"Sözleşme Türü  *(Orijinal: {cust_row['contract_type']})*",
        categorical_options["contract_type"], "val_contract_type", str(cust_row["contract_type"]),
    )
    sim_company_size = select_row(
        f"Firma Büyüklüğü  *(Orijinal: {cust_row['company_size']})*",
        categorical_options["company_size"], "val_company_size", str(cust_row["company_size"]),
    )
    sim_channel = select_row(
        f"Kayıt Kanalı  *(Orijinal: {cust_row['signup_channel']})*",
        categorical_options["signup_channel"], "val_signup_channel", str(cust_row["signup_channel"]),
    )
    sim_charges = slider_row(
        f"Aylık Ücret ($)  *(Orijinal: ${cust_row['monthly_charges']:.2f})*",
        15.0, 400.0, "val_monthly_charges", float(cust_row["monthly_charges"]),
    )
    sim_last_login = slider_row(
        f"Son Girişten Geçen Gün  *(Orijinal: {cust_row['days_since_last_login']})*",
        0, 30, "val_days_since_last_login", int(cust_row["days_since_last_login"]),
    )
    sim_freq = slider_row(
        f"Aylık Aktif Gün  *(Orijinal: {cust_row['usage_frequency']})*",
        1, 30, "val_usage_frequency", int(cust_row["usage_frequency"]),
    )
    sim_adoption = slider_row(
        f"Özellik Kullanım Oranı (%)  *(Orijinal: {adoption_display})*",
        5, 100, "val_feature_adoption_rate", safe_int(cust_row["feature_adoption_rate"], median_adoption),
        help_text="Orijinal veri eksikse popülasyon medyanı varsayılan olarak kullanılır.",
    )
    sim_tickets = number_row(
        f"Destek Kaydı  *(Orijinal: {cust_row['support_tickets']})*",
        0, 20, "val_support_tickets", int(cust_row["support_tickets"]),
    )
    sim_delay = number_row(
        f"Fatura Gecikme Günü  *(Orijinal: {cust_row['payment_delay_days']})*",
        0, 60, "val_payment_delay_days", int(cust_row["payment_delay_days"]),
    )
    sim_nps = slider_row(
        f"NPS Skoru  *(Orijinal: {nps_display})*",
        -100, 100, "val_nps_score", safe_int(cust_row["nps_score"], median_nps),
        help_text="Orijinal veri eksikse popülasyon medyanı varsayılan olarak kullanılır.",
    )

with col_right:
    st.subheader("📈 4. Güncel Sonuç & SHAP Açıklanabilirlik")

    sim_raw = pd.DataFrame(
        [{
            "tenure_months": sim_tenure,
            "monthly_charges": sim_charges,
            "days_since_last_login": sim_last_login,
            "usage_frequency": sim_freq,
            "feature_adoption_rate": sim_adoption,
            "support_tickets": sim_tickets,
            "payment_delay_days": sim_delay,
            "nps_score": sim_nps,
            "contract_type": sim_contract,
            "company_size": sim_company_size,
            "signup_channel": sim_channel,
        }]
    )
    sim_encoded = pd.get_dummies(sim_raw, columns=CATEGORICAL_COLS, drop_first=True)
    for col in model_features:
        if col not in sim_encoded.columns:
            sim_encoded[col] = 0
    sim_input = sim_encoded[model_features]

    sim_prob = float(model.predict_proba(sim_input)[0][1])
    sim_pred = int(model.predict(sim_input)[0])
    prob_diff = (sim_prob - original_prob) * 100

    if sim_pred == 1:
        st.error(f"🚨 **Güncel Durum:** Terk Edebilir (Risk: %{sim_prob * 100:.1f})")
    else:
        st.success(f"✅ **Güncel Durum:** Şirkette Kalacak (Risk: %{sim_prob * 100:.1f})")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric(
            label="Simüle Edilen Risk",
            value=f"%{sim_prob * 100:.1f}",
            delta=f"{prob_diff:+.1f}% Değişim",
            delta_color="inverse",
        )
    with m_col2:
        st.metric(
            label="Orijinal Risk",
            value=f"%{original_prob * 100:.1f}",
            delta="Yüksek Risk" if original_prob >= 0.5 else "Düşük Risk",
            delta_color="off",
        )

    st.progress(sim_prob)

    # SHAP Analizi
    st.markdown("#### 🧠 Model Neden Bu Kararı Verdi? (SHAP Analizi)")
    shap_values = explainer(sim_input)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

# ==============================================================================
# BÖLÜM 5: YAZILI KÖK NEDEN RAPORU & KURTARMA REÇETESİ
# ==============================================================================
st.markdown("---")
st.subheader("📄 5. Yönetici / Müşteri Müdahale Raporu")

shap_df = pd.DataFrame({"feature": model_features, "importance": shap_values[0].values})
risk_drivers = shap_df[shap_df["importance"] > 0].sort_values(by="importance", ascending=False)

feature_tr_names = {
    "days_since_last_login": "Son Girişten Geçen Gün",
    "payment_delay_days": "Fatura Gecikme Günü",
    "support_tickets": "Destek Kaydı Sayısı",
    "contract_type_Monthly": "Aylık Sözleşme Türü",
    "monthly_charges": "Aylık Hizmet Bedeli",
    "feature_adoption_rate": "Özellik Kullanım Oranı",
    "tenure_months": "Abonelik Süresi",
    "usage_frequency": "Aylık Aktif Kullanım Günü",
    "nps_score": "NPS Skoru",
    "company_size_Medium": "Firma Büyüklüğü (Orta)",
    "company_size_Small": "Firma Büyüklüğü (Küçük)",
    "signup_channel_Organic": "Kayıt Kanalı (Organik)",
    "signup_channel_Paid": "Kayıt Kanalı (Ücretli)",
    "signup_channel_Sales-Led": "Kayıt Kanalı (Satış Odaklı)",
}

original_values = {
    "contract_type": cust_row["contract_type"], "company_size": cust_row["company_size"],
    "signup_channel": cust_row["signup_channel"], "monthly_charges": float(cust_row["monthly_charges"]),
    "days_since_last_login": int(cust_row["days_since_last_login"]), "usage_frequency": int(cust_row["usage_frequency"]),
    "feature_adoption_rate": safe_int(cust_row["feature_adoption_rate"], median_adoption),
    "support_tickets": int(cust_row["support_tickets"]), "payment_delay_days": int(cust_row["payment_delay_days"]),
    "nps_score": safe_int(cust_row["nps_score"], median_nps), "tenure_months": int(cust_row["tenure_months"]),
}
sim_values = {
    "contract_type": sim_contract, "company_size": sim_company_size, "signup_channel": sim_channel,
    "monthly_charges": sim_charges, "days_since_last_login": sim_last_login, "usage_frequency": sim_freq,
    "feature_adoption_rate": sim_adoption, "support_tickets": sim_tickets, "payment_delay_days": sim_delay,
    "nps_score": sim_nps, "tenure_months": sim_tenure,
}
field_labels_tr = {
    "contract_type": "Sözleşme Türü", "company_size": "Firma Büyüklüğü", "signup_channel": "Kayıt Kanalı",
    "monthly_charges": "Aylık Ücret", "days_since_last_login": "Son Giriş", "usage_frequency": "Aktif Kullanım",
    "feature_adoption_rate": "Özellik Kullanımı", "support_tickets": "Destek Kaydı",
    "payment_delay_days": "Fatura Gecikmesi", "nps_score": "NPS Skoru", "tenure_months": "Abonelik Süresi",
}

applied_changes = [
    f"• {field_labels_tr[k]}: {original_values[k]} ➔ {sim_values[k]}"
    for k in original_values if original_values[k] != sim_values[k]
]
changes_text = "\n  ".join(applied_changes) if applied_changes else "• Henüz bir simülasyon müdahalesi yapılmadı (Orijinal veriler geçerli)."

top_risk_reasons = [
    f"• {feature_tr_names.get(r['feature'], r['feature'])}: Riski yukarı çeken kritik faktör (+{r['importance']:.2f} etki puanı)."
    for _, r in risk_drivers.head(3).iterrows()
]
reasons_text = "\n  ".join(top_risk_reasons) if top_risk_reasons else "• Belirgin bir risk artırıcı faktör bulunmuyor."

primary_risk_feature = risk_drivers.iloc[0]["feature"] if len(risk_drivers) > 0 else ""
action_notes = {
    "days_since_last_login": "Müşterinin sisteme girişi uzun süredir kesilmiş durumda. CSM ekibi acilen bir 'Re-activation & Check-in' araması yapmalı ve yeni ürün özellikleri tanıtılmalıdır.",
    "payment_delay_days": "Temel risk fatura gecikmelerinden kaynaklanıyor. Finans/Satış ekibi müşteriye özel esnek ödeme planı veya %10 indirimle Yıllık Sözleşme teklifi iletmelidir.",
    "support_tickets": "Müşteri yüksek teknik destek talebi sebebiyle memnuniyetsizlik yaşıyor. Kıdemli bir destek mühendisi açık biletleri öncelikli olarak çözmek için iletişime geçmelidir.",
    "contract_type_Monthly": "Müşterinin aylık sözleşmede olması kayıp riskini artırıyor. Yıllık plana geçiş için kampanya sunularak müşteri uzun vadeli güvenceye alınmalıdır.",
    "nps_score": "Müşterinin memnuniyet algısı düşük. Yönetici seviyesinde bir 'Voice of Customer' görüşmesi planlanmalı ve şikayetler önceliklendirilmelidir.",
}
action_note = action_notes.get(
    primary_risk_feature,
    "Müşteri profilini korumak adına düzenli hesap sağlığı kontrolü (Health Check) yapılmalı ve ürün adaptasyon oranı artırılmalıdır.",
)

col_rep_btn, _ = st.columns([1, 3])
with col_rep_btn:
    generate_report = st.button("📋 Kurtarma Planı Raporu Üret", width="stretch", type="primary")

if generate_report:
    report_output = f"""
================================================================================
📋 CHURNLENS YÖNETİCİ & MÜŞTERİ MÜDAHALE RAPORU | MÜŞTERİ #{selected_customer_id}
================================================================================

1. MEVCUT DURUM & FİNANSAL PROFİL:
  - Orijinal Risk Skoru : %{original_prob * 100:.1f} ({'🚨 YÜKSEK RİSK (Terk Edebilir)' if original_prob >= 0.5 else '✅ GÜVENLİ'})
  - Aylık Getiri (MRR)  : ${cust_row['monthly_charges']:.2f} (Yıllık Ciro Değeri: ${(float(cust_row['monthly_charges']) * 12):,.2f})
  - Sözleşme / Süre     : {cust_row['contract_type']} / {cust_row['tenure_months']} Ay
  - Firma / Kanal       : {cust_row['company_size']} / {cust_row['signup_channel']}

2. MODEL TEŞHİSİ: NEDEN RİSK ALTINDA? (Kök Neden Analizi):
  {reasons_text}

3. UYGULANAN SİMÜLASYON MÜDAHALELERİ:
  {changes_text}
  ➔ Sonuç: Risk %{original_prob * 100:.1f} seviyesinden %{sim_prob * 100:.1f} seviyesine çekildi ({prob_diff:+.1f}% değişim).

4. ÖNERİLEN KURTARMA AKSİYONU (Playbook Notu):
  "{action_note}"
================================================================================
"""
    st.code(report_output, language="text")
    st.download_button(
        "⬇️ Raporu .txt Olarak İndir",
        data=report_output.encode("utf-8"),
        file_name=f"churnlens_rapor_musteri_{selected_customer_id}.txt",
        mime="text/plain",
    )