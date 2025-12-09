import cv2
import numpy as np
import os
from src.processamento import preprocessar_clahe
from src.utils import detectar_borda, salvar_csv


def contar_colonias(
    img_path,
    minRadius=5,
    maxRadius=40,
    minDist=20,
    param2=18,
    clipLimit=2.0,
    diametro_real_mm=None,
    salvar_imagem=True,
    salvar_csv_flag=True
):
    # ---------------------------
    # 1) Carregar imagem
    # ---------------------------
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {img_path}")

    img_gray, img_clahe = preprocessar_clahe(img, clipLimit=clipLimit)

    # ---------------------------
    # 2) Detectar borda da placa
    # ---------------------------
    cx, cy, r = detectar_borda(img_gray)

    # Criar máscara da placa
    mask = np.zeros_like(img_gray, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), int(r * 0.88), 255, -1)

    # Remover bordas reflexivas
    mask = cv2.erode(mask, np.ones((15,15), np.uint8))


    # Aplicar máscara
    masked = cv2.bitwise_and(img_clahe, img_clahe, mask=mask)

    # ---------------------------
    # 3) Calcular escala mm/pixel
    # ---------------------------
    scale_mm = None
    if diametro_real_mm is not None:
        diametro_pixels = 2 * r
        scale_mm = diametro_real_mm / diametro_pixels

    # ---------------------------
    # 4) Detectar colônias (círculos)
    # ---------------------------
    circles = cv2.HoughCircles(
    masked,
    cv2.HOUGH_GRADIENT,
    dp=1.4,
    minDist=25,
    param1=80,
    param2=22,   # aumenta para reduzir falsos positivos
    minRadius=5, # colônias pequenas
    maxRadius=18 # limite superior realista
    )


    img_marked = img.copy()
    registros = []
    count = 0

    # ---------------------------
    # 5) Processar círculos detectados
    # ---------------------------
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        h, w = mask.shape

        for (x, y, rad) in circles:

            # Ignorar valores fora da imagem (corrige IndexError)
            if x < 0 or y < 0 or x >= w or y >= h:
                continue

            # Ignorar pontos fora da placa
            if mask[y, x] == 0:
                continue

            count += 1

            # Desenhar círculos
            cv2.circle(img_marked, (x, y), rad, (0, 0, 255), 2)
            cv2.circle(img_marked, (x, y), 2, (0, 255, 0), 3)

            # Registrar dados
            registro = {
                "x_px": x,
                "y_px": y,
                "r_px": rad
            }

            if scale_mm is not None:
                registro["x_mm"] = round(x * scale_mm, 3)
                registro["y_mm"] = round(y * scale_mm, 3)
                registro["r_mm"] = round(rad * scale_mm, 3)

            registros.append(registro)

    # ---------------------------
    # 6) Salvar imagem marcada
    # ---------------------------
    nome_base = os.path.basename(img_path).split(".")[0]

    saida_img = None
    if salvar_imagem:
        saida_img = f"resultados/imagens_marcadas/{nome_base}_marcada.png"
        os.makedirs(os.path.dirname(saida_img), exist_ok=True)
        cv2.imwrite(saida_img, img_marked)

    # ---------------------------
    # 7) Salvar CSV com registros
    # ---------------------------
    saida_csv = None
    if salvar_csv_flag:
        saida_csv = f"resultados/csv/{nome_base}.csv"
        salvar_csv(saida_csv, registros)

    # ---------------------------
    # 8) Retorno organizado
    # ---------------------------
    return {
        "quantidade": count,
        "img_saida": saida_img,
        "csv_saida": saida_csv,
        "scale_mm": scale_mm,
        "registros": registros
    }
