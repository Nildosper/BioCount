import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import zipfile
from PIL import Image

st.set_page_config(layout="wide")
st.title("🧫 BioCount — Contagem Automatizada com Placa de Referência")

# =========================
# Funções auxiliares
# =========================
def load_image(file):
    img = Image.open(file).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def preprocess_difference(img_ref, img_sample, clipLimit):
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_sam = cv2.cvtColor(img_sample, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    gray_ref = clahe.apply(gray_ref)
    gray_sam = clahe.apply(gray_sam)

    return cv2.absdiff(gray_sam, gray_ref)

def segment_colonies(diff):
    _, th = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    return th

def extract_metrics(binary, original, px_to_mm):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    h, w = original.shape[:2]
    cxp, cyp = w // 2, h // 2
    raio_placa = min(cxp, cyp) * 0.92  # remove borda

    registros = []
    overlay = original.copy()

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 40:
            continue

        cx, cy = centroids[i]
        dist = np.sqrt((cx - cxp) ** 2 + (cy - cyp) ** 2)
        if dist > raio_placa:
            continue

        r_eq = np.sqrt(area / np.pi)

        registros.append({
            "x_px": int(cx),
            "y_px": int(cy),
            "area_px2": area,
            "diametro_mm": 2 * r_eq * px_to_mm
        })

        cv2.circle(overlay, (int(cx), int(cy)), int(r_eq), (0, 255, 0), 2)

    return registros, overlay

# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ Parâmetros")
clipLimit = st.sidebar.slider("CLAHE clipLimit", 1.0, 5.0, 1.5)
diametro_placa_mm = st.sidebar.number_input("Diâmetro real da placa (mm)", 90)

# =========================
# PROCESSAMENTO EM LOTE
# =========================
st.header("📁 Processamento em lote")

ref_file = st.file_uploader(
    "Imagem de referência (obrigatória)",
    type=["jpg", "png"]
)

files = st.file_uploader(
    "Imagens das amostras",
    type=["jpg", "png"],
    accept_multiple_files=True
)

if ref_file and files:
    img_ref = load_image(ref_file)
    px_to_mm = diametro_placa_mm / img_ref.shape[0]

    imagens_processadas = []
    resultados = {}

    for file in files:
        img = load_image(file)

        if img.shape != img_ref.shape:
            img = cv2.resize(img, (img_ref.shape[1], img_ref.shape[0]))

        diff = preprocess_difference(img_ref, img, clipLimit)
        binary = segment_colonies(diff)
        registros, overlay = extract_metrics(binary, img, px_to_mm)

        df = pd.DataFrame(registros)

        resultados[file.name] = {
            "contagem": len(df),
            "diametro_medio_mm": df["diametro_mm"].mean() if not df.empty else 0
        }

        imagens_processadas.append((file.name, overlay, diff))

    # ===== Mostrar imagens (3 por linha) =====
    st.subheader("🖼️ Imagens processadas")
    cols = st.columns(3)

    for i, (nome, overlay, diff) in enumerate(imagens_processadas):
        with cols[i % 3]:
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=nome)

    # ===== Resultados =====
    st.subheader("📄 Resultados")
    df_final = pd.DataFrame(resultados)
    st.dataframe(df_final)

    # ===== ZIP =====
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = f"{tmp}/resultados.csv"
        df_final.to_csv(csv_path)

        zip_path = f"{tmp}/resultados.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.write(csv_path, "resultados.csv")
            for nome, overlay, diff in imagens_processadas:
                p = f"{tmp}/{nome}_processado.png"
                cv2.imwrite(p, overlay)
                z.write(p, f"imagens/{nome}_processado.png")

        with open(zip_path, "rb") as f:
            st.download_button("⬇️ Baixar resultados (ZIP)", f, "resultados.zip")

else:
    st.info("Envie a imagem de referência e pelo menos uma amostra.")
