import cv2

def preprocessar_clahe(img, clipLimit=2.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
    img_clahe = clahe.apply(gray)
    return gray, img_clahe
