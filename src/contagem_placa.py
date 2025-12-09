import cv2
import numpy as np
import pandas as pd
import tempfile

def contar_colonias(
    img_input,
    minRadius=5,
    maxRadius=30,
    minDist=20,
    param2=18,
    clipLimit=2.0,
    diametro_real_mm=90
):

    # ---------------------------------------------------------
    # Aceitar tanto caminho quanto matriz numpy
    # ---------------------------------------------------------
    if isinstance(img_input, str):
        img_bgr = cv2.imread(img_input)
    else:
        img_bgr = img_input

    if img_bgr is None:
        raise ValueError("❌ Erro: imagem não pôde ser carregada.")

    original = img_bgr.copy()

    # ---------------------------------------------------------
    # Pré-processamento
    # ---------------------------------------------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=minDist,
        param1=50,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    registros = []

    # ---------------------------------------------------------
    # Desenhar círculos e salvar registros
    # ---------------------------------------------------------
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            registros.append({
                "x": int(x),
                "y": int(y),
                "r_px": int(r)
            })

            cv2.circle(original, (x, y), r, (0, 255, 0), 2)
            cv2.circle(original, (x, y), 2, (0, 0, 255), 3)

    # ---------------------------------------------------------
    # Criar CSV temporário para download
    # ---------------------------------------------------------
    df = pd.DataFrame(registros)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as tmp_csv:
        csv_path = tmp_csv.name
        df.to_csv(csv_path, index=False, sep=";")

    # ---------------------------------------------------------
    # Converter imagem BGR → RGB para funcionar no Streamlit
    # ---------------------------------------------------------
    img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # Retorno final
    # ---------------------------------------------------------
    return {
        "quantidade": len(registros),
        "registros": registros,
        "img_saida": img_rgb,   # pronto para st.image()
        "csv_saida": csv_path,  # pronto para download
    }
