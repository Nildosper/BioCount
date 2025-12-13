import cv2
import numpy as np
from PIL import Image

def load_image(file):
    img = Image.open(file).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def criar_mascara_placa(img, fator=0.90):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    raio = int(min(cx, cy) * fator)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), raio, 255, -1)
    return mask
