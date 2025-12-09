import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import plotly.express as px
from src.contagem_placa import contar_colonias
from src.lote import processar_lote

st.title("🧫 BioCount – Contagem Automática de Colônias")

st.sidebar.header("⚙ Parâmetros de Detecção")

minRadius = st.sidebar.slider("Raio mínimo", 3, 20, 5)
maxRadius = st.sidebar.slider("Raio máximo", 10, 60, 30)
minDist = st.sidebar.slider("Distância mínima entre centros", 10, 60, 20)
param2 = st.sidebar.slider("Param2 (Hough)", 8, 50, 18)
clipLimit = st.sidebar.slider("CLAHE clipLimit", 1.0, 10.0, 2.0)

diametro_real_mm = st.sidebar.number_input(
    "Diâmetro real da placa (mm)", value=90
)

st.subheader("📤 Carregue uma imagem da placa:")

img_file = st.file_uploader(
    "Escolha uma imagem de placa (jpg/png)", type=["jpg","png"]
)

if img_file is not None:
    img = Image.open(img_file)
    img_np = np.array(img)

    # salvar temporário
    tmp = "temp_input.png"
    cv2.imwrite(tmp, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    # rodar processamento
    result = contar_colonias(
        tmp,
        minRadius=minRadius,
        maxRadius=maxRadius,
        minDist=minDist,
        param2=param2,
        clipLimit=clipLimit,
        diametro_real_mm=diametro_real_mm
    )

    st.subheader(f"🔢 Colônias detectadas: **{result['quantidade']}**")

    st.image(result["img_saida"], caption="Imagem processada", use_column_width=True)

    # tabela
    df = pd.DataFrame(result["registros"])
    st.write("📄 Coordenadas e raios detectados:")
    st.dataframe(df)

    # gráfico distribuição de raios
    if "r_px" in df.columns:
        fig = px.histogram(df, x="r_px", nbins=20, title="Distribuição dos raios detectados")
        st.plotly_chart(fig)

    # download CSV
    with open(result["csv_saida"], "rb") as f:
        st.download_button(
            "📥 Baixar CSV das colônias",
            f,
            file_name="colônias.csv"
        )

# ------------------------------
# PROCESSAMENTO EM LOTE
# ------------------------------

st.subheader("📁 Processar várias imagens (Lote)")
pasta = st.text_input("Digite o caminho da pasta:")

if st.button("Processar lote"):
    if pasta:
        lote = processar_lote(pasta, diametro_mm=diametro_real_mm)
        df_lote = pd.DataFrame(lote, columns=["arquivo","colônias"])
        st.write(df_lote)

        fig = px.bar(df_lote, x="arquivo", y="colônias", title="Resultado por imagem")
        st.plotly_chart(fig)
