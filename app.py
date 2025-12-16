import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.decomposition import PCA

# ================================
# KONFIGURASI HALAMAN
# ================================
st.set_page_config(
    page_title="Lung Cancer Clustering",
    layout="wide"
)

st.title("🫁 Lung Cancer Clustering")
st.caption("Metode: K-Means & Logistic Regression")

# ================================
# UPLOAD DATASET
# ================================
uploaded_file = st.file_uploader(
    "Upload Dataset Lung Cancer (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Silakan upload dataset CSV terlebih dahulu")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("Dataset berhasil dimuat")

st.write("Jumlah data:", df.shape[0])
st.dataframe(df.head())

# ================================
# PREPROCESSING
# ================================
st.subheader("⚙️ Preprocessing Data")

df_proc = df.copy()
label_encoders = {}

for col in df_proc.columns:
    if df_proc[col].dtype == "object":
        le = LabelEncoder()
        df_proc[col] = le.fit_transform(df_proc[col])
        label_encoders[col] = le

X = df_proc

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.success("Encoding & Standardisasi selesai")

# ================================
# K-MEANS CLUSTERING
# ================================
st.subheader("🔹 K-Means Clustering")

k = st.slider("Jumlah Cluster (K)", 2, 6, 3)

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=20
)

clusters = kmeans.fit_predict(X_scaled)
df["Cluster"] = clusters

sil_score = silhouette_score(X_scaled, clusters)
st.metric("Silhouette Score", round(sil_score, 3))

# ================================
# VISUALISASI CLUSTER (STREAMLIT NATIVE)
# ================================
st.subheader("📊 Visualisasi Cluster (PCA 2D)")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

viz_df = pd.DataFrame({
    "PCA1": X_pca[:, 0],
    "PCA2": X_pca[:, 1],
    "Cluster": clusters
})

st.scatter_chart(
    viz_df,
    x="PCA1",
    y="PCA2",
    color="Cluster"
)

# ================================
# LOGISTIC REGRESSION
# ================================
st.subheader("🔹 Logistic Regression (Prediksi Cluster)")

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

st.metric("Akurasi Logistic Regression", round(acc, 3))

# ================================
# INPUT DATA BARU
# ================================
st.subheader("🧪 Input Data Pasien Baru")

input_data = {}

with st.form("input_form"):
    for col in df_proc.columns:
        if col in label_encoders:
            input_data[col] = st.selectbox(
                col,
                label_encoders[col].classes_
            )
        else:
            input_data[col] = st.number_input(
                col,
                float(df[col].min()),
                float(df[col].max())
            )

    submit = st.form_submit_button("Prediksi Cluster")

if submit:
    input_df = pd.DataFrame([input_data])

    for col, le in label_encoders.items():
        input_df[col] = le.transform(input_df[col])

    input_scaled = scaler.transform(input_df)
    predicted_cluster = kmeans.predict(input_scaled)[0]

    st.success(f"Pasien termasuk ke **Cluster {predicted_cluster}**")

# ================================
# DATA AKHIR
# ================================
st.subheader("📄 Contoh Data dengan Cluster")
st.dataframe(df.head(10))
