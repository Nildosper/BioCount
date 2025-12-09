import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import plotly.express as px
from src.contagem_placa import contar_colonias

st.title("🧫 BioCount – Contagem Automática de Colônias")

# ============================
# SIDEBAR – PARÂMETROS
# ============================

st.sidebar.header("⚙ Parâmetros de Detecção")

minRadius = st.sidebar.slider("Raio mínimo", 3, 20, 5)
maxRadius = st.sidebar.slider("Raio máximo", 10, 60, 30)
minDist = st.sidebar.slider("Distância mínima entre centros", 10, 60, 20)
param2 = st.sidebar.slider("Param2 (Hough)", 8, 50, 18)
clipLimit = st.sidebar.slider("CLAHE clipLimit", 1.0, 10.0, 2.0)

diametro_real_mm = st.sidebar.number_input(
    "Diâmetro real da placa (mm)", value=90
)

# ============================
# MODO TABS
# ============================

tab1, tab2 = st.tabs(["🖼️ Imagem única", "📁 Processamento em lote"])

# ============================
# TAB 1 — IMAGEM ÚNICA
# ============================

with tab1:

    st.subheader("📤 Carregue uma imagem da placa:")

    img_file = st.file_uploader(
        "Escolha uma imagem (jpg/png)", type=["jpg", "png"]
    )

    if img_file is not None:
        img = Image.open(img_file)
        img_np = np.array(img)

        # converter para BGR para o OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # processar
        result = contar_colonias(
            img_bgr,
            minRadius=minRadius,
            maxRadius=maxRadius,
            minDist=minDist,
            param2=param2,
            clipLimit=clipLimit,
            diametro_real_mm=diametro_real_mm
        )

        st.subheader(f"🔢 Colônias detectadas: **{result['quantidade']}**")

        st.image(
            cv2.cvtColor(result["img_saida"], cv2.COLOR_BGR2RGB),
            caption="Imagem processada",
            use_column_width=True
        )

        # tabela
        df = pd.DataFrame(result["registros"])
        st.write("📄 Coordenadas e raios detectados:")
        st.dataframe(df)

        # histograma
        if "r_px" in df.columns:
            fig = px.histogram(df, x="r_px", nbins=20, title="Distribuição dos raios detectados")
            st.plotly_chart(fig)


# ============================
# TAB 2 — PROCESSAMENTO EM LOTE
# ============================

with tab2:

    st.subheader("📁 Envie várias imagens de uma só vez:")

    uploaded_files = st.file_uploader(
        "Selecione várias imagens (jpg/png)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:

        resultados = []

        for arquivo in uploaded_files:
            img = Image.open(arquivo)
            img_np = np.array(img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            result = contar_colonias(
                img_bgr,
                minRadius=minRadius,
                maxRadius=maxRadius,
                diametro_real_mm=diametro_real_mm,
                minDist=minDist,
                param2=param2,
                clipLimit=clipLimit
            )

            resultados.append({
                "arquivo": arquivo.name,
                "quantidade": result["quantidade"],
                "img": result["img_saida"]
            })

        # mostrar resultados
        df_lote = pd.DataFrame(resultados)[["arquivo", "quantidade"]]
        st.write(df_lote)

        # gráfico
        fig = px.bar(df_lote, x="arquivo", y="quantidade", title="Colônias por imagem")
        st.plotly_chart(fig)

        # exibir imagens marcadas
        st.subheader("🖼️ Imagens processadas")

        for r in resultados:
            st.image(
                cv2.cvtColor(r["img"], cv2.COLOR_BGR2RGB),
                caption=f"{r['arquivo']} — {r['quantidade']} colônias"
            )
