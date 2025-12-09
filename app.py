import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import plotly.express as px
import tempfile
from src.contagem_placa import contar_colonias

st.set_page_config(page_title="BioCount", layout="wide")

st.title("🧫 BioCount – Contagem Automática de Colônias")

# ===============================================================
# SIDEBAR – parâmetros de detecção
# ===============================================================
st.sidebar.header("⚙ Parâmetros de Detecção")

minRadius = st.sidebar.slider("Raio mínimo", 3, 20, 5)
maxRadius = st.sidebar.slider("Raio máximo", 10, 60, 30)
minDist = st.sidebar.slider("Distância mínima entre centros", 10, 60, 20)
param2 = st.sidebar.slider("Param2 (Hough)", 8, 50, 18)
clipLimit = st.sidebar.slider("CLAHE clipLimit", 1.0, 10.0, 2.0)

diametro_real_mm = st.sidebar.number_input(
    "Diâmetro real da placa (mm)", value=90
)

# ===============================================================
# ABAS
# ===============================================================
tab1, tab2 = st.tabs(["🖼 Imagem única", "📁 Processamento em lote"])

# ===============================================================
#  TAB 1 — IMAGEM ÚNICA
# ===============================================================
with tab1:
    st.subheader("📤 Carregue uma imagem da placa:")

    img_file = st.file_uploader("Escolha uma imagem (jpg/png)", type=["jpg", "png"])

    if img_file is not None:

        # abrir a imagem
        img = Image.open(img_file)
        img_np = np.array(img)

        # criar arquivo temporário compatível com streamlit cloud
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        # processar
        result = contar_colonias(
            temp_path,
            minRadius=minRadius,
            maxRadius=maxRadius,
            minDist=minDist,
            param2=param2,
            clipLimit=clipLimit,
            diametro_real_mm=diametro_real_mm
        )

        st.subheader(f"🔢 Colônias detectadas: **{result['quantidade']}**")
        st.image(result["img_saida"], caption="Imagem processada", use_column_width=True)

        df = pd.DataFrame(result["registros"])
        st.write("📄 Coordenadas e raios detectados:")
        st.dataframe(df)

        # Histograma
        if "r_px" in df.columns:
            fig = px.histogram(df, x="r_px", nbins=20,
                               title="Distribuição dos raios detectados")
            st.plotly_chart(fig)

        # botão de download CSV
        with open(result["csv_saida"], "rb") as f:
            st.download_button(
                "📥 Baixar CSV das colônias",
                f,
                file_name="colonias.csv"
            )

# ===============================================================
#  TAB 2 — PROCESSAMENTO EM LOTE
# ===============================================================
with tab2:
    st.subheader("📁 Envie várias imagens de uma só vez:")

    uploaded_files = st.file_uploader(
        "Selecione várias imagens (jpg/png)",
        type=["jpg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        resultados_lote = []
        imagens_processadas = []

        for file in uploaded_files:
            img = Image.open(file)
            img_np = np.array(img)

            # criar temp para cada imagem
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                temp_path = tmp_img.name
                cv2.imwrite(temp_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

            # processar
            result = contar_colonias(
                temp_path,
                minRadius=minRadius,
                maxRadius=maxRadius,
                minDist=minDist,
                param2=param2,
                clipLimit=clipLimit,
                diametro_real_mm=diametro_real_mm
            )

            resultados_lote.append([file.name, result["quantidade"]])
            imagens_processadas.append((file.name, result))

        st.success("Processamento em lote concluído!")

        # tabela
        df_lote = pd.DataFrame(resultados_lote, columns=["Arquivo", "Colônias"])
        st.write("📄 Resultado por imagem:")
        st.dataframe(df_lote)

        # gráfico
        fig = px.bar(df_lote, x="Arquivo", y="Colônias",
                     title="Quantidade de colônias por imagem")
        st.plotly_chart(fig)

        # exibição individual
        st.subheader("🖼 Imagens processadas:")

        for nome, result in imagens_processadas:
            st.image(result["img_saida"],
                     caption=f"{nome} — {result['quantidade']} colônias",
                     use_column_width=True)

            # download CSV por imagem
            with open(result["csv_saida"], "rb") as f:
                st.download_button(
                    f"📥 Baixar CSV — {nome}",
                    f,
                    file_name=f"{nome}_colonias.csv"
                )
