import streamlit as st
import pandas as pd
import cv2
import zipfile
import tempfile
from src.utils import load_image, criar_mascara_placa
from src.processamento import processar_amostra

st.set_page_config(layout="wide")
st.title("🧫 BioCount — Processamento em Lote")

st.sidebar.header("⚙️ Parâmetros")
clipLimit = st.sidebar.slider("CLAHE clipLimit", 1.0, 5.0, 1.5)
diametro_placa_mm = st.sidebar.number_input("Diâmetro real da placa (mm)", 90)

st.subheader("Imagem de referência (obrigatória)")
ref_file = st.file_uploader("Upload da placa sem colônias", type=["jpg","png"])

st.subheader("Imagens das amostras")
files = st.file_uploader(
    "Upload das placas com colônias",
    type=["jpg","png"],
    accept_multiple_files=True
)

if ref_file and files:
    img_ref = load_image(ref_file)
    mask_placa = criar_mascara_placa(img_ref)
    px_to_mm = diametro_placa_mm / img_ref.shape[0]

    resultados = {}
    imagens = []

    for file in files:
        img_sam = load_image(file)
        registros, overlay, diff = processar_amostra(
            img_ref, img_sam, mask_placa, clipLimit, px_to_mm
        )

        df = pd.DataFrame(registros)

        resultados[file.name] = {
            "contagem": len(df),
            "diametro_medio_mm": df["diametro_mm"].mean() if not df.empty else 0
        }

        imagens.append((file.name, overlay, diff))

    st.subheader("📸 Resultados visuais")
    cols = st.columns(3)
    for i, (nome, overlay, _) in enumerate(imagens):
        with cols[i % 3]:
            st.image(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                caption=nome,
                use_container_width=True
            )

    df_final = pd.DataFrame(resultados)
    st.subheader("📄 Resultados consolidados")
    st.dataframe(df_final)

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = f"{tmp}/resultados.csv"
        df_final.to_csv(csv_path)

        zip_path = f"{tmp}/resultados.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.write(csv_path, "resultados.csv")
            for nome, overlay, diff in imagens:
                p = f"{tmp}/{nome}_processado.png"
                cv2.imwrite(p, overlay)
                z.write(p, f"imagens/{nome}_processado.png")

        with open(zip_path, "rb") as f:
            st.download_button(
                "⬇️ Baixar resultados (ZIP)",
                f,
                "resultados.zip"
            )
else:
    st.warning("⚠️ Envie a imagem de referência e pelo menos uma amostra.")
