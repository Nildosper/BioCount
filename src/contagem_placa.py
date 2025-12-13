import cv2
import numpy as np
import pandas as pd
import tempfile

def contar_colonias(
    img_input,
    minRadius=5,
    maxRadius=25,
    minDist=25,
    param2=20,
    clipLimit=2.0,
    diametro_real_mm=90
):

    # ============================================================
    # 1) CARREGAMENTO
    # ============================================================
    if isinstance(img_input, np.ndarray):
        img = img_input.copy()
    elif isinstance(img_input, str):
        img = cv2.imread(img_input)
    else:
        raise TypeError("Entrada inválida em contar_colonias().")

    if img is None:
        raise ValueError("Erro ao carregar imagem.")

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ============================================================
    # 2) MELHOR PREPROCESSAMENTO
    # ============================================================
    # Remover ruído
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # Equalização adaptativa
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # ============================================================
    # 3) DETECTAR APENAS A REGIÃO ÚTIL (CÍRCULO DA PLACA)
    # ============================================================
    h, w = gray.shape
    cx, cy = w//2, h//2
    raio_placa = min(cx, cy) - 30  # margem maior para evitar borda metálica

    mascara_placa = np.zeros_like(gray)
    cv2.circle(mascara_placa, (cx, cy), raio_placa, 255, -1)

    gray_masked = cv2.bitwise_and(gray, mascara_placa)

    # ============================================================
    # 4) HOUGHCIRCLES OTIMIZADO
    # ============================================================
    circles = cv2.HoughCircles(
        gray_masked,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=minDist,
        param1=60,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    registros = []

    if circles is not None:
        circles = np.round(circles[0]).astype("int")

        for (x, y, r) in circles:

            # ⛔ 1. Ignorar pontos fora da placa (por segurança)
            if np.sqrt((x - cx)**2 + (y - cy)**2) > raio_placa:
                continue

            # Criar máscara para medir brilho local
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), r, 255, -1)
            intensidade = cv2.mean(gray, mask=mask)[0]

            # ⛔ 2. Remover reflexos BRANCOS (muito brilhantes)
            if intensidade > 180:
                continue

            # ⛔ 3. Remover pontos extremamente escuros (sujeira)
            if intensidade < 30:
                continue

            # ⛔ 4. Remover círculos muito grandes (reflexos)
            if r > maxRadius * 0.85:
                continue

            # MANTER círculo válido
            registros.append({
                "x": int(x),
                "y": int(y),
                "r_px": int(r),
                "intensidade": round(intensidade, 2)
            })

            # Desenhar
            cv2.circle(original, (x, y), r, (0, 255, 0), 2)
            cv2.circle(original, (x, y), 2, (0, 0, 255), 3)

    # ============================================================
    # 5) SALVAR CSV TEMPORÁRIO
    # ============================================================
    df = pd.DataFrame(registros)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as f:
        df.to_csv(f.name, index=False, sep=";")
        csv_path = f.name

    img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    return {
        "quantidade": len(registros),
        "registros": registros,
        "img_saida": img_rgb,
        "csv_saida": csv_path,
    }
