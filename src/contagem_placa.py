import cv2
import numpy as np

def extract_metrics(binary, original, px_to_mm, mask_placa):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    registros = []
    overlay = original.copy()

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 30:
            continue

        cx, cy = centroids[i]
        cx, cy = int(cx), int(cy)

        if mask_placa[cy, cx] == 0:
            continue

        raio_eq = np.sqrt(area / np.pi)
        diam_mm = 2 * raio_eq * px_to_mm

        registros.append({
            "x_px": cx,
            "y_px": cy,
            "area_px2": area,
            "raio_eq_px": raio_eq,
            "diametro_mm": diam_mm
        })

        cv2.circle(overlay, (cx, cy), int(raio_eq), (0,255,0), 2)

    return registros, overlay

