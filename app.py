import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, accuracy_score

# ================================
# KONFIGURASI HALAMAN
# ================================
st.set_page_config(
    page_title="Lung Cancer Clustering",
    layout="wide"
)

st.title("🫁 Lung Cancer Clustering & Prediction")
st.caption("K-Means Clustering + Logistic Regression")

# ================================
# UPLOAD DATASET
# ================================
st.subheader("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload file CSV (Lung Cancer Dataset)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("⚠️ Silakan upload dataset CSV terlebih dahulu")
    st.stop()

df = pd.read_csv(uploaded_file)

st.success("✅ Dataset berhasil dimuat")
st.write("Jumlah data:", df.shape[0])
st.dataframe(df.head())

# ================================
# PREPROCESSING
# ================================
st.subheader("⚙️ Preprocessing")

df_processed = df.copy()

label_encoders = {}
for col in df_processed.columns:
    if df_processed[col].dtype == "object":
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le

X = df_processed

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.success("✔️ Encoding & Standardisasi selesai")

# ================================
# CLUSTERING
# ================================
st.subheader("🔹 K-Means Clustering")

k = st.slider("Pilih jumlah cluster (K)", 2, 6, 3)

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)
clusters = kmeans.fit_predict(X_scaled)
df["Cluster"] = clusters

sil_score = silhouette_score(X_scaled, clusters)
st.write("📌 **Silhouette Score:**", round(sil_score, 3))

# ================================
# LOGISTIC REGRESSION
# ================================
st.subheader("🔹 Logistic Regression")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    df["Cluster"],
    test_size=0.2,
    random_state=42
)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write("📌 **Akurasi Logistic Regression:**", round(acc, 3))

# ================================
# INPUT DATA BARU (PREDIKSI CLUSTER)
# ================================
st.subheader("🧪 Input Data Pasien Baru")

input_data = {}

with st.form("input_form"):
    for col in df_processed.columns:
        if col in label_encoders:
            options = label_encoders[col].classes_
            input_data[col] = st.selectbox(col, options)
        else:
            input_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()))

    submit = st.form_submit_button("🔍 Prediksi Cluster")

if submit:
    input_df = pd.DataFrame([input_data])

    # Encode input
    for col, le in label_encoders.items():
        input_df[col] = le.transform(input_df[col])

    # Scaling
    input_scaled = scaler.transform(input_df)

    # Prediksi cluster
    predicted_cluster = kmeans.predict(input_scaled)[0]

    st.success(f"✅ Pasien ini termasuk ke **Cluster {predicted_cluster}**")

# ================================
# HASIL DATA
# ================================
st.subheader("📊 Data dengan Cluster")
st.dataframe(df.head(10))
