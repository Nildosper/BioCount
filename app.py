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

def extract_metrics(binary, original, px_to_mm, max_radius_factor=0.8):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    registros = []
    overlay = original.copy()

    h, w = original.shape[:2]
    radius_placa = min(w, h) / 2

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 30:
            continue

        cx, cy = centroids[i]
        radius_eq = np.sqrt(area / np.pi)

        # Filtrando as colônias perto da borda
        if np.sqrt((cx - w/2)**2 + (cy - h/2)**2) > radius_placa * max_radius_factor:
            continue

        # Circularidade
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
tab1, tab2 = st.tabs(["📁 Processamento em lote"])

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

        resultados_colunas = {}
        imagens_zip = []

        for file in files:
            img_sam = load_image(file)

            # 🔒 Garantir mesma escala
            if img_sam.shape != img_ref.shape:
                img_sam = cv2.resize(
                    img_sam,
                    (img_ref.shape[1], img_ref.shape[0]),
                    interpolation=cv2.INTER_AREA
                )

            diff = preprocess_difference(img_ref, img_sam, clipLimit)

            # Kernel adaptativo
            k = max(3, img_ref.shape[0] // 300)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            th = segment_colonies(diff)

            registros, overlay = extract_metrics(th, img_sam, px_to_mm)

            df = pd.DataFrame(registros)

            # 📊 Métricas agregadas
            resultados_colunas[file.name] = {
                "contagem_total": len(df),
                "area_media_px2": df["area_px2"].mean() if not df.empty else 0,
                "diametro_medio_mm": df["diametro_mm"].mean() if not df.empty else 0,
                "desvio_diam_mm": df["diametro_mm"].std() if not df.empty else 0,
                "densidade_col_cm2": len(df) / (np.pi * (diametro_placa_mm/20)**2)
            }

            # Mostrar imagem
            st.image(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                caption=f"Processado: {file.name}",
                use_column_width=True
            )

            imagens_zip.append((file.name, overlay, diff))

        # CSV final (amostras em colunas)
        df_final = pd.DataFrame(resultados_colunas)
        st.subheader("📄 Resultados consolidados")
        st.dataframe(df_final)

        # 📦 ZIP
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = f"{tmp}/resultados_lote.csv"
            df_final.to_csv(csv_path)

            zip_path = f"{tmp}/resultados_lote.zip"
            with zipfile.ZipFile(zip_path, "w") as z:
                z.write(csv_path, "resultados_lote.csv")

                for nome, overlay, diff in imagens_zip:
                    p1 = f"{tmp}/{nome}_processado.png"
                    p2 = f"{tmp}/{nome}_diferenca.png"
                    cv2.imwrite(p1, overlay)
                    cv2.imwrite(p2, diff)
                    z.write(p1, f"imagens/{nome}_processado.png")
                    z.write(p2, f"imagens/{nome}_diferenca.png")

            with open(zip_path, "rb") as f:
                st.download_button(
                    "⬇️ Baixar todos os resultados (ZIP)",
                    f,
                    "resultados_lote.zip"
                )
