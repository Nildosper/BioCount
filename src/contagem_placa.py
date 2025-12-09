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

    # Se vier arquivo do Streamlit (numpy array)
    if isinstance(img_input, np.ndarray):
        img = img_input.copy()

    # Se vier caminho de arquivo
    elif isinstance(img_input, str):
        img = cv2.imread(img_input)

    else:
        raise TypeError("Entrada inválida para contar_colonias().")

    if img is None:
        raise ValueError("Erro ao carregar imagem.")

    # Conversão para cópia trabalhável
    original = img.copy()

    # ---- PRÉ-PROCESSAMENTO ----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # ---- DETECÇÃO ----
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

    if circles is not None:
        circles = np.round(circles[0]).astype("int")

        for (x, y, r) in circles:
            registros.append({"x": x, "y": y, "r_px": r})
            cv2.circle(original, (x, y), r, (0, 255, 0), 2)
            cv2.circle(original, (x, y), 2, (0, 0, 255), 3)

    # ---- SALVAR CSV TEMPORÁRIO ----
    df = pd.DataFrame(registros)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as f:
        df.to_csv(f.name, index=False, sep=";")
        csv_path = f.name

    # ---- CONVERTER PARA RGB ----
    img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    return {
        "quantidade": len(registros),
        "registros": registros,
        "img_saida": img_rgb,
        "csv_saida": csv_path,
    }
