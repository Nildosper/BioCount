import cv2
import numpy as np
import csv

def detectar_borda(img_gray):
    # Reduz ruído grande
    blur = cv2.GaussianBlur(img_gray, (15, 15), 0)

    # Canny
    edges = cv2.Canny(blur, 50, 120)

    # Hough para círculos grandes
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.3,
        minDist=500,
        param1=80,
        param2=40,
        minRadius=int(min(img_gray.shape) * 0.35),
        maxRadius=int(min(img_gray.shape) * 0.50)
    )

    h, w = img_gray.shape

    if circles is None:
        # fallback: assume placa centralizada
        return w // 2, h // 2, int(min(h, w) * 0.40)

    circles = np.round(circles[0]).astype("int")

    # escolher o círculo mais central
    def score(c):
        x, y, r = c
        return abs(x - w//2) + abs(y - h//2)

    x, y, r = sorted(circles, key=score)[0]
    return x, y, r



def salvar_csv(caminho, registros):
    import os
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    if not registros:
        return
    keys = registros[0].keys()
    with open(caminho, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(registros)
