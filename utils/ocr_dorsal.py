"""
ocr_dorsal.py — OCR para detección de números de dorsal en imágenes de ciclistas.

Usa Tesseract con modo dígitos-solamente y múltiples variantes de
preprocesamiento (CLAHE + Otsu, threshold adaptivo, invertido) para
maximizar la tasa de detección en condiciones de carrera real.

Dependencias del sistema:
    sudo apt install tesseract-ocr
    pip install pytesseract
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import pytesseract
    # Verificar que el binario está disponible
    pytesseract.get_tesseract_version()
    _TESSERACT_OK = True
    logger.info("Tesseract OCR disponible — dorsal OCR activado")
except Exception as _e:
    _TESSERACT_OK = False
    logger.warning(
        "Tesseract no disponible (%s). "
        "Instalar: sudo apt install tesseract-ocr && pip install pytesseract",
        _e,
    )

# Configuración: solo dígitos, bloque de texto simple, motor LSTM
_TSR_CFG = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"


# ─── API pública ──────────────────────────────────────────────────────────────

def detectar_dorsal(img: np.ndarray) -> Optional[str]:
    """
    Detecta el número de dorsal en la imagen BGR del ciclista.

    Estrategia:
        1. Recorta la región del torso (30 %–85 % de altura) donde suele
           estar el número de carrera.
        2. Genera múltiples variantes preprocesadas para cubrir distintas
           condiciones de iluminación y color de fondo.
        3. Vota por el candidato más frecuente entre todas las variantes.

    Args:
        img: Frame BGR (recorte del ciclista, cualquier tamaño).

    Returns:
        Número de dorsal como string (ej. ``"42"``), o ``None`` si no
        se detectó ninguno con suficiente confianza.
    """
    if not _TESSERACT_OK or img is None or img.size == 0:
        return None

    try:
        h, w = img.shape[:2]

        # ── Recorte del torso ─────────────────────────────────────────────────
        y0 = int(h * 0.28)
        y1 = int(h * 0.82)
        torso = img[y0:y1, :] if y1 > y0 else img

        candidatos: list[str] = []
        for variante in _preparar_variantes(torso):
            texto = pytesseract.image_to_string(variante, config=_TSR_CFG).strip()
            nums  = re.findall(r"\b\d{1,4}\b", texto)
            candidatos.extend(nums)

        if not candidatos:
            return None

        # ── Voto por mayoría ──────────────────────────────────────────────────
        freq: dict[str, int] = {}
        for n in candidatos:
            freq[n] = freq.get(n, 0) + 1
        mejor = max(freq, key=lambda n: (freq[n], len(n)))
        logger.info("OCR dorsal: %s  (freq=%s)", mejor, freq)
        return mejor

    except Exception as exc:
        logger.warning("Error OCR: %s", exc)
        return None


# ─── Helpers privados ─────────────────────────────────────────────────────────

def _preparar_variantes(img: np.ndarray) -> list[np.ndarray]:
    """
    Devuelve hasta 4 imágenes en escala de grises preprocesadas para
    maximizar la detección de Tesseract con distintos fondos de dorsal.
    """
    h, w = img.shape[:2]

    # Escalar a mínimo 300 px de alto para mejor resolución OCR
    escala = max(1.0, 300.0 / h)
    if escala > 1.0:
        img = cv2.resize(
            img, (int(w * escala), int(h * escala)),
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE + Otsu (bueno para bajo contraste)
    clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, otsu  = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Threshold adaptivo gaussiano (bueno para iluminación no uniforme)
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8,
    )

    # 3. Invertido Otsu (dorsal claro sobre fondo oscuro)
    inv = cv2.bitwise_not(otsu)

    # 4. Sharpening + Otsu (bordes más definidos)
    kernel  = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp   = cv2.filter2D(gray, -1, kernel)
    _, otsu2 = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return [otsu, adapt, inv, otsu2]
