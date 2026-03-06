"""
crop_worker.py — Worker de recorte y guardado de capturas para PC Linux.

Reemplaza MaliCropWorker (Mali-G610 OpenCL) del Orange Pi por operaciones
cv2 en CPU. El recorte en GPU no es necesario en una PC con CUDA ya que el
cuello de botella es la inferencia YOLO, no el encode JPEG.

Interfaz idéntica a mali_crop.py para compatibilidad con main_timing.py.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─── Mensajes inter-hilo ──────────────────────────────────────────────────────

@dataclass
class CropJob:
    """Solicitud de recorte enviada por el detector de cruce al worker."""
    frame:             np.ndarray
    track_id:          int
    bbox_xyxy:         tuple[int, int, int, int]
    ts_ns:             int
    con_casco:         bool
    dorsal:            str           = "N/A"
    tiempo_carrera_ms: Optional[int] = None


@dataclass
class DBRecord:
    """Registro listo para ser persistido en SQLite por el AsyncDBWriter."""
    track_id:          int
    ts_ns:             int
    con_casco:         bool
    dorsal:            str
    foto_meta_path:    Optional[str]
    tiempo_carrera_ms: Optional[int]       = None
    dorsal_ocr:        Optional[str]       = None
    crop_np:           Optional[np.ndarray] = None  # para OCR asíncrono


# ─── Worker ───────────────────────────────────────────────────────────────────

class CropWorker(threading.Thread):
    """
    Hilo daemon que consume CropJob, recorta en CPU y pone DBRecord en db_q.

    Separado en hilo propio para que el encode JPEG y la escritura a disco
    no bloqueen el loop de inferencia CUDA.
    """

    _SENTINEL    = object()
    MAX_SIDE     = 1280
    JPEG_QUALITY = 97

    def __init__(
        self,
        crop_q:   queue.Queue,
        db_q:     queue.Queue,
        save_dir: str | Path = "data/captures",
    ) -> None:
        super().__init__(name="CropWorker", daemon=True)
        self._crop_q   = crop_q
        self._db_q     = db_q
        self._save_dir = Path(save_dir)

    def run(self) -> None:
        logger.info("CropWorker iniciado (CPU JPEG encode)")

        while True:
            try:
                job = self._crop_q.get(timeout=1.0)
            except queue.Empty:
                continue

            if job is self._SENTINEL:
                logger.info("CropWorker: parada recibida")
                break

            foto_path: Optional[str]        = None
            crop_np:   Optional[np.ndarray] = None
            try:
                foto_path, crop_np = self._crop_and_save(job)
                logger.debug("Captura: track=%d → %s", job.track_id, foto_path)
            except Exception:
                logger.exception("Error en crop para track=%d", job.track_id)

            # Enviar inmediatamente — el OCR corre en hilo separado en main_timing
            self._db_q.put(DBRecord(
                track_id          = job.track_id,
                ts_ns             = job.ts_ns,
                con_casco         = job.con_casco,
                dorsal            = job.dorsal,
                foto_meta_path    = foto_path,
                tiempo_carrera_ms = job.tiempo_carrera_ms,
                crop_np           = crop_np,
            ))

    def _crop_and_save(self, job: CropJob) -> tuple[str, np.ndarray]:
        """Recorta, mejora nitidez, guarda JPEG y devuelve (ruta, array)."""
        x1, y1, x2, y2 = job.bbox_xyxy
        h, w = job.frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"BBox invalida: ({x1},{y1},{x2},{y2})")

        crop = job.frame[y1:y2, x1:x2].copy()

        # Sharpening para foto de meta más nítida
        kernel = np.array([[0, -0.5, 0],
                            [-0.5, 3, -0.5],
                            [0, -0.5, 0]], dtype=np.float32)
        crop = cv2.filter2D(crop, -1, kernel)

        # Redimensionar si supera MAX_SIDE
        ch, cw = crop.shape[:2]
        if max(ch, cw) > self.MAX_SIDE:
            scale  = self.MAX_SIDE / max(ch, cw)
            target = (int(cw * scale), int(ch * scale))
            crop   = cv2.resize(crop, target, interpolation=cv2.INTER_LANCZOS4)

        ok, buf = cv2.imencode(".jpg", crop,
                               [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
        if not ok:
            raise RuntimeError("cv2.imencode fallo")

        ts_ms    = job.ts_ns // 1_000_000
        date_str = datetime.now().strftime("%Y%m%d")
        out_path = self._save_dir / date_str / f"t{job.track_id:04d}_{ts_ms}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(buf.tobytes())
        return str(out_path), crop

    def stop(self) -> None:
        """Señal de parada al worker (no bloquea)."""
        self._crop_q.put(self._SENTINEL)
