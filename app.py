import cv2
import numpy as np

def contar_colonias(
    img_bgr,
    minRadius=5,
    maxRadius=30,
    minDist=20,
    param2=18,
    clipLimit=2.0,
    diametro_real_mm=90
):

    if isinstance(img_bgr, str):
        img_bgr = cv2.imread(img_bgr)

    if img_bgr is None:
        raise ValueError("Erro: imagem não pôde ser carregada.")

    original = img_bgr.copy()

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

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        h, w = gray.shape
        cx, cy = w//2, h//2  # centro estimado da placa
        raio_placa = min(cx, cy) - 20  # margem de segurança

        for (x, y, r) in circles:

            # ⛔ 1. Filtrar círculos fora da área da placa
            if np.sqrt((x - cx)**2 + (y - cy)**2) > raio_placa:
                continue

            # ⛔ 2. Filtrar reflexos muito intensos (área média > 200)
            mascara = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mascara, (x, y), r, 255, -1)
            media_int = cv2.mean(gray, mask=mascara)[0]
            if media_int > 200:
                continue

            # ⛔ 3. Filtrar círculos muito grandes (reflexos brancos enormes)
            if r > maxRadius * 0.9:
                continue

            registros.append({
                "x": int(x),
                "y": int(y),
                "r_px": int(r),
                "intensidade": round(media_int, 2)
            })

            cv2.circle(original, (x, y), r, (0, 255, 0), 2)
            cv2.circle(original, (x, y), 2, (0, 0, 255), 3)

    resultado = {
        "quantidade": len(registros),
        "registros": registros,
        "img_saida": original
    }

    return resultado
