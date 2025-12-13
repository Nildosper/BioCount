import cv2
import numpy as np
from .contagem_placa import extract_metrics

def processar_amostra(img_ref, img_sam, mask_placa, clipLimit, px_to_mm):

    if img_sam.shape != img_ref.shape:
        img_sam = cv2.resize(img_sam, (img_ref.shape[1], img_ref.shape[0]))

    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_sam = cv2.cvtColor(img_sam, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
    gray_ref = clahe.apply(gray_ref)
    gray_sam = clahe.apply(gray_sam)

    diff = cv2.absdiff(gray_sam, gray_ref)
    diff = cv2.bitwise_and(diff, diff, mask=mask_placa)

    _, th = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = max(3, img_ref.shape[0] // 300)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, 2)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, 2)

    registros, overlay = extract_metrics(th, img_sam, px_to_mm, mask_placa)

    return registros, overlay, diff
