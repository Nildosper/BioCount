import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import tempfile

st.set_page_config(page_title="BioCount", layout="wide")

st.title("🧫 BioCount – Contagem de Colônias com Imagem de Referência")

# =============================
# Função principal com referência
# =============================
def contar_colonias_com_referencia(
    img_ref_bgr,
    img_sample_bgr,
    minRadius,
    maxRadius,
    minDist,
    param2,
    clipLimit
):

    # Pré-processamento
    gray_ref = cv2.cvtColor(img_ref_bgr, cv2.COLOR_BGR2GRAY)
    gray_smp = cv2.cvtColor(img_sample_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    gray_ref = clahe.apply(gray_ref)
    gray_smp = clahe.apply(gray_smp)

    # Subtração absoluta
    diff = cv2.absdiff(gray_smp, gray_ref)

    # Detecção de círculos
    circles = cv2.HoughCircles(
        diff,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=minDist,
        param1=50,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    output = img_sample_bgr.copy()
    registros = []

    if circles is not None:
        circles = np.round(circles[0]).astype(int)

        h, w = diff.shape
        cx, cy = w // 2, h // 2
        raio_placa = min(cx, cy) - 20

        for (x, y, r) in circles:

            # Filtro espacial (fora da placa)
            if np.sqrt((x - cx)**2 + (y - cy)**2) > raio_placa:
                continue

            registros.append({"x": x, "y": y, "r_px": r})

            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

    return {
        "quantidade": len(registros),
        "registros": registros,
        "img_saida": cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
        "diff": diff
    }

# =============================
# Parâmetros
# =============================
st.sidebar.header("⚙️ Parâmetros de Detecção")

minRadius = st.sidebar.slider("Raio mínimo", 3, 30, 6)
maxRadius = st.sidebar.slider("Raio máximo", 5, 60, 21)
minDist = st.sidebar.slider("Distância mínima entre centros", 5, 50, 21)
param2 = st.sidebar.slider("Param2 (Hough)", 5, 50, 21)
clipLimit = st.sidebar.slider("CLAHE clipLimit", 0.5, 5.0, 1.4)

modo = st.radio("Modo de processamento", ["Imagem única", "Processamento em lote"])

# =============================
# IMAGEM ÚNICA
# =============================
if modo == "Imagem única":

    st.subheader("📌 Passo 1 – Envie a imagem de referência (SEM colônias)")
    ref_file = st.file_uploader(
        "Imagem de referência",
        type=["jpg", "jpeg", "png"],
        key="ref_unica"
    )

    st.subheader("📌 Passo 2 – Envie a imagem da amostra (COM colônias)")
    sample_file = st.file_uploader(
        "Imagem da amostra",
        type=["jpg", "jpeg", "png"],
        key="sample_unica"
    )

    if ref_file and sample_file:

        ref_img = cv2.cvtColor(np.array(Image.open(ref_file)), cv2.COLOR_RGB2BGR)
        sample_img = cv2.cvtColor(np.array(Image.open(sample_file)), cv2.COLOR_RGB2BGR)

        resultado = contar_colonias_com_referencia(
            ref_img,
            sample_img,
            minRadius,
            maxRadius,
            minDist,
            param2,
            clipLimit
        )

        st.success(f"🧪 Colônias detectadas: {resultado['quantidade']}")

        col1, col2 = st.columns(2)
        with col1:
            st.image(resultado["img_saida"], caption="Imagem processada", width=400)
        with col2:
            st.image(resultado["diff"], caption="Imagem diferença (amostra - referência)", width=400)

        df = pd.DataFrame(resultado["registros"])
        st.dataframe(df)

    else:
        st.warning("⚠️ É obrigatório enviar a imagem de referência e a imagem da amostra.")

# =============================
# PROCESSAMENTO EM LOTE
# =============================
else:

    st.subheader("📌 Passo 1 – Envie a imagem de referência (SEM colônias)")
    ref_file = st.file_uploader(
        "Imagem de referência",
        type=["jpg", "jpeg", "png"],
        key="ref_lote"
    )

    st.subheader("📌 Passo 2 – Envie as imagens das amostras")
    sample_files = st.file_uploader(
        "Imagens das amostras",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="sample_lote"
    )

    if ref_file and sample_files:

        ref_img = cv2.cvtColor(np.array(Image.open(ref_file)), cv2.COLOR_RGB2BGR)

        resultados = []

        for file in sample_files:
            sample_img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_RGB2BGR)

            res = contar_colonias_com_referencia(
                ref_img,
                sample_img,
                minRadius,
                maxRadius,
                minDist,
                param2,
                clipLimit
            )

            resultados.append({
                "arquivo": file.name,
                "quantidade": res["quantidade"]
            })

        df = pd.DataFrame(resultados)
        st.success("✅ Processamento em lote concluído")
        st.dataframe(df)

    else:
        st.warning("⚠️ Para processar em lote, a imagem de referência é obrigatória.")
