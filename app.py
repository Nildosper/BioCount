import streamlit as st
import numpy as np
import cv2
import pandas as pd
import tempfile
import zipfile
import plotly.express as px
from PIL import Image
from pathlib import Path

# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def detectar_placa(gray):
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=1000,
        param1=50,
        param2=30,
        minRadius=int(gray.shape[0]*0.35),
        maxRadius=int(gray.shape[0]*0.48)
    )
    if circles is None:
        raise ValueError("Placa não detectada na imagem de referência.")
    x, y, r = circles[0][0]
    return int(x), int(y), int(r)


def criar_mascara(shape, cx, cy, r):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), int(r*0.97), 255, -1)
    return mask


def contar_colonias(img_bgr, img_ref_gray, mask,
                    minRadius, maxRadius, minDist, param2, clipLimit):

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    diff = cv2.absdiff(gray, img_ref_gray)
    diff = cv2.bitwise_and(diff, diff, mask=mask)

    _, thresh = cv2.threshold(diff, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    circles = cv2.HoughCircles(
        thresh,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=minDist,
        param1=50,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    registros = []
    output = img_bgr.copy()

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for x, y, r in circles:
            area = np.pi * r * r
            perimetro = 2 * np.pi * r
            circularidade = (4 * np.pi * area) / (perimetro**2)

            registros.append({
                "x": x,
                "y": y,
                "raio_px": r,
                "area_px2": area,
                "circularidade": circularidade
            })

            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 2)

    return registros, output, diff


# =========================================================
# INTERFACE
# =========================================================

st.set_page_config(layout="wide")
st.title("🧫 BioCount – Contagem Automática de Colônias")

st.sidebar.header("⚙ Parâmetros de Detecção")

minRadius = st.sidebar.slider("Raio mínimo (px)", 3, 20, 6)
maxRadius = st.sidebar.slider("Raio máximo (px)", 10, 60, 25)
minDist = st.sidebar.slider("Distância mínima entre centros", 10, 60, 20)
param2 = st.sidebar.slider("Sensibilidade Hough (param2)", 8, 50, 20)
clipLimit = st.sidebar.slider("CLAHE clipLimit", 1.0, 5.0, 1.5)

tabs = st.tabs(["🧪 Processamento único", "📦 Processamento em lote"])

# =========================================================
# PROCESSAMENTO ÚNICO
# =========================================================

with tabs[0]:
    st.subheader("1️⃣ Imagem de referência (SEM colônias)")
    ref_file = st.file_uploader("Imagem de referência", type=["jpg","png"], key="ref1")

    st.subheader("2️⃣ Imagem da amostra")
    img_file = st.file_uploader("Imagem da amostra", type=["jpg","png"], key="sam1")

    if ref_file and img_file:
        ref = np.array(Image.open(ref_file).convert("RGB"))
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)

        cx, cy, r = detectar_placa(ref_gray)
        mask = criar_mascara(ref_gray.shape, cx, cy, r)

        img = np.array(Image.open(img_file).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        registros, out, diff = contar_colonias(
            img_bgr, ref_gray, mask,
            minRadius, maxRadius, minDist, param2, clipLimit
        )

        df = pd.DataFrame(registros)

        st.markdown(f"### 🔢 Colônias detectadas: **{len(df)}**")

        c1, c2 = st.columns(2)
        with c1:
            st.image(out, caption="Imagem processada", use_container_width=True)
        with c2:
            st.image(diff, caption="Diferença (amostra - referência)", use_container_width=True)

        if not df.empty:
            st.subheader("📊 Distribuição de tamanho")
            fig = px.histogram(df, x="raio_px", nbins=20)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.scatter(df, x="area_px2", y="circularidade")
            st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(df)

            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Baixar CSV", csv, "resultado_unico.csv")


# =========================================================
# PROCESSAMENTO EM LOTE
# =========================================================

with tabs[1]:
    st.subheader("1️⃣ Imagem de referência (OBRIGATÓRIA)")
    ref_file = st.file_uploader("Imagem de referência", type=["jpg","png"], key="ref2")

    st.subheader("2️⃣ Imagens das amostras")
    imgs = st.file_uploader("Selecione várias imagens",
                             type=["jpg","png"],
                             accept_multiple_files=True)

    if ref_file and imgs:
        ref = np.array(Image.open(ref_file).convert("RGB"))
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)

        cx, cy, r = detectar_placa(ref_gray)
        mask = criar_mascara(ref_gray.shape, cx, cy, r)

        resultados = {}
        zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")

        with zipfile.ZipFile(zip_buffer.name, "w") as zipf:

            for img_file in imgs:
                img = np.array(Image.open(img_file).convert("RGB"))
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                registros, out, diff = contar_colonias(
                    img_bgr, ref_gray, mask,
                    minRadius, maxRadius, minDist, param2, clipLimit
                )

                df = pd.DataFrame(registros)
                resultados[img_file.name] = df.shape[0]

                # salvar imagens
                out_path = f"{img_file.name}_processada.png"
                diff_path = f"{img_file.name}_diff.png"
                csv_path = f"{img_file.name}.csv"

                cv2.imwrite(out_path, out)
                cv2.imwrite(diff_path, diff)

                zipf.write(out_path)
                zipf.write(diff_path)

                df.to_csv(csv_path, index=False)
                zipf.write(csv_path)

        df_res = pd.DataFrame.from_dict(resultados, orient="index", columns=["colônias"])
        st.dataframe(df_res)

        st.download_button(
            "📦 Baixar todos os resultados",
            open(zip_buffer.name, "rb"),
            "resultados_biocount.zip"
        )
