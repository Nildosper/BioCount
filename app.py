import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import zipfile
import plotly.express as px
from PIL import Image

st.set_page_config(layout="wide")
st.title("🧫 BioCount — Contagem Automatizada com Placa de Referência")

# -------------------------
# Funções auxiliares
# -------------------------
def load_image(file):
    img = Image.open(file).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def preprocess_difference(img_ref, img_sample, clipLimit):
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_sam = cv2.cvtColor(img_sample, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
    gray_ref = clahe.apply(gray_ref)
    gray_sam = clahe.apply(gray_sam)

    diff = cv2.absdiff(gray_sam, gray_ref)
    return diff

def segment_colonies(diff):
    _, th = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    return th

def extract_metrics(binary, original, px_to_mm):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    registros = []
    overlay = original.copy()

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 30:
            continue

        cx, cy = centroids[i]
        radius_eq = np.sqrt(area / np.pi)

        circ = (4 * np.pi * area) / (stats[i, cv2.CC_STAT_WIDTH]**2 + 1e-6)

        registros.append({
            "x_px": int(cx),
            "y_px": int(cy),
            "area_px2": area,
            "raio_eq_px": radius_eq,
            "diametro_mm": 2 * radius_eq * px_to_mm,
            "circularidade": circ
        })

        cv2.circle(
            overlay,
            (int(cx), int(cy)),
            int(radius_eq),
            (0,255,0),
            2
        )

    return registros, overlay

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("⚙️ Parâmetros")
clipLimit = st.sidebar.slider("CLAHE clipLimit", 1.0, 5.0, 1.5)
diametro_placa_mm = st.sidebar.number_input("Diâmetro real da placa (mm)", 90)

# -------------------------
# Abas
# -------------------------
tab1, tab2 = st.tabs(["📷 Processamento único", "📁 Processamento em lote"])

# =====================================================
# PROCESSAMENTO ÚNICO
# =====================================================
with tab1:
    st.subheader("Imagem de referência (sem colônias)")
    ref_file = st.file_uploader("Upload da referência", type=["jpg","png"], key="ref1")

    st.subheader("Imagem da amostra")
    sample_file = st.file_uploader("Upload da amostra", type=["jpg","png"], key="sam1")

    if ref_file and sample_file:
        img_ref = load_image(ref_file)
        img_sam = load_image(sample_file)

        diff = preprocess_difference(img_ref, img_sam, clipLimit)
        binary = segment_colonies(diff)

        px_to_mm = diametro_placa_mm / img_ref.shape[0]

        registros, overlay = extract_metrics(binary, img_sam, px_to_mm)
        df = pd.DataFrame(registros)

        col1, col2 = st.columns(2)
        col1.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Imagem processada")
        col2.image(diff, caption="Diferença (amostra - referência)", clamp=True)

        st.metric("Colônias detectadas", len(df))

        st.subheader("📊 Distribuição de tamanhos")
        fig = px.histogram(df, x="diametro_mm", nbins=20)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📄 Dados")
        st.dataframe(df)

        # Downloads
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = f"{tmp}/resultado.csv"
            img_path = f"{tmp}/imagem_processada.png"
            diff_path = f"{tmp}/diferenca.png"

            df.to_csv(csv_path, index=False)
            cv2.imwrite(img_path, overlay)
            cv2.imwrite(diff_path, diff)

            zip_path = f"{tmp}/resultados.zip"
            with zipfile.ZipFile(zip_path, "w") as z:
                z.write(csv_path, "resultado.csv")
                z.write(img_path, "imagem_processada.png")
                z.write(diff_path, "imagem_diferenca.png")

            with open(zip_path, "rb") as f:
                st.download_button("⬇️ Baixar resultados (ZIP)", f, "resultados.zip")

    else:
        st.warning("⚠️ Envie a imagem de referência e a amostra para processar.")

# =====================================================
# PROCESSAMENTO EM LOTE
# =====================================================
with tab2:
    st.subheader("Imagem de referência (obrigatória)")
    ref_file = st.file_uploader("Upload da referência", type=["jpg","png"], key="ref2")

    st.subheader("Imagens das amostras")
    files = st.file_uploader(
        "Upload das amostras",
        type=["jpg","png"],
        accept_multiple_files=True
    )

    if ref_file and files:
        img_ref = load_image(ref_file)
        px_to_mm = diametro_placa_mm / img_ref.shape[0]

        resultados = []

        for file in files:
            img_sam = load_image(file)
            diff = preprocess_difference(img_ref, img_sam, clipLimit)
            binary = segment_colonies(diff)
            regs, overlay = extract_metrics(binary, img_sam, px_to_mm)

            for r in regs:
                r["arquivo"] = file.name
                resultados.append(r)

            st.image(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                caption=f"Processado: {file.name}"
            )

        df = pd.DataFrame(resultados)

        st.subheader("Resumo do lote")
        st.dataframe(df)

        fig = px.histogram(df, x="diametro_mm", color="arquivo", nbins=20)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ Referência e pelo menos uma amostra são obrigatórias.")
