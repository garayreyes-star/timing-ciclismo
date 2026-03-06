"""
detector.py — Detector YOLO + ByteTrack + ReID para PC Linux x86_64 con CUDA.

Reemplaza NPUEngine (Zhouyi) + SimpleTracker de Orange Pi por:
  - YOLO ONNX via onnxruntime-gpu (CUDAExecutionProvider) con fallback a CPU
  - ByteTrack via supervision library
  - ReID por histograma HSV (cosine similarity, threshold 0.65)
  - Detección de casco: con modelo COCO estándar → False por defecto;
    con modelo custom que incluya clase 'helmet' → activar HELMET_CLASS_ID

Modelo recomendado: yolov8n.onnx (exportar con ultralytics):
    from ultralytics import YOLO
    YOLO("yolov8n.pt").export(format="onnx", imgsz=640, opset=12)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── Clases COCO relevantes ───────────────────────────────────────────────────
_PERSON_CLASS  = 0
_BICYCLE_CLASS = 1

# Si tu modelo tiene clase de casco, cambia a su ID; con COCO estándar = None
HELMET_CLASS_ID: Optional[int] = None

# ─── ReID ─────────────────────────────────────────────────────────────────────
REID_THRESHOLD   = 0.65
REID_GALLERY_TTL = 30.0   # segundos


# ─── Track ────────────────────────────────────────────────────────────────────

@dataclass
class Track:
    """Track activo de un ciclista (interfaz compatible con Orange Pi port)."""
    track_id:  int
    bbox_xyxy: tuple[int, int, int, int]
    y_centro:  float    # Y del centro del bbox para DetectorCruce
    con_casco: bool     # True si se detectó casco (requiere modelo custom)
    conf:      float = 0.0


# ─── YOLO Detector ────────────────────────────────────────────────────────────

class YOLODetector:
    """
    YOLO ONNX via onnxruntime-gpu.
    Compatible con YOLOv8n/YOLOv10n/YOLO11n exportados a ONNX COCO 80 clases.
    CUDAExecutionProvider con fallback automático a CPU.
    """

    _INPUT_SIZE = 640

    def __init__(
        self,
        model_path:     str,
        conf_thr:       float       = 0.35,
        iou_thr:        float       = 0.45,
        target_classes: list[int]  | None = None,
    ) -> None:
        import onnxruntime as ort

        self._conf_thr = conf_thr
        self._iou_thr  = iou_thr
        self._target   = set(target_classes) if target_classes else {_PERSON_CLASS}

        providers = []
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {"device_id": 0}))
            logger.info("YOLODetector: CUDAExecutionProvider (GTX 1650)")
        else:
            logger.warning("YOLODetector: CUDA no disponible → CPU")
        providers.append("CPUExecutionProvider")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess       = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        self._input_name = self._sess.get_inputs()[0].name
        self._provider   = self._sess.get_providers()[0]

        inp = self._sess.get_inputs()[0]
        logger.info("Modelo: %s  entrada=%s  provider=%s",
                    model_path, inp.shape, self._provider)

    @property
    def provider(self) -> str:
        return self._provider

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detecta en frame BGR.
        Retorna lista de {class_id, conf, bbox_xyxy: (x1,y1,x2,y2)}.
        """
        blob, ratio, dw, dh = self._preprocess(frame)
        raw = self._sess.run(None, {self._input_name: blob})[0]
        return self._postprocess(raw, ratio, dw, dh, frame.shape[:2])

    def _preprocess(self, frame: np.ndarray):
        """Letterbox a 640×640, BGR→RGB, normalizar, NCHW float32."""
        h, w = frame.shape[:2]
        S     = self._INPUT_SIZE
        ratio = min(S / h, S / w)
        nh, nw = int(h * ratio), int(w * ratio)

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas  = np.full((S, S, 3), 114, dtype=np.uint8)
        dh      = (S - nh) // 2
        dw      = (S - nw) // 2
        canvas[dh:dh + nh, dw:dw + nw] = resized

        blob = canvas[..., ::-1].astype(np.float32) / 255.0   # BGR→RGB, [0,1]
        blob = blob.transpose(2, 0, 1)[np.newaxis]              # HWC→NCHW
        return blob, ratio, dw, dh

    def _postprocess(
        self,
        raw:   np.ndarray,
        ratio: float,
        dw:    int,
        dh:    int,
        orig:  tuple[int, int],
    ) -> list[dict]:
        """
        raw: [1, 84, 8400] (YOLOv8) → transpose → [8400, 84]
        Columnas: cx, cy, w, h, score_cls0..score_cls79
        """
        orig_h, orig_w = orig
        preds = raw[0].T                               # [8400, 84]

        class_scores = preds[:, 4:]                    # [8400, 80]
        class_ids    = class_scores.argmax(axis=1)
        confs        = class_scores.max(axis=1)

        # Filtrar por confianza y clase objetivo
        mask    = (confs >= self._conf_thr) & np.isin(class_ids, list(self._target))
        preds   = preds[mask]
        confs   = confs[mask]
        class_ids = class_ids[mask]

        if len(preds) == 0:
            return []

        # cx,cy,w,h → x1,y1,x2,y2 en espacio letterbox
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Deshacer letterbox
        x1 = ((x1 - dw) / ratio).clip(0, orig_w)
        y1 = ((y1 - dh) / ratio).clip(0, orig_h)
        x2 = ((x2 - dw) / ratio).clip(0, orig_w)
        y2 = ((y2 - dh) / ratio).clip(0, orig_h)

        # NMS por clase
        boxes  = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        result = []
        for cls in np.unique(class_ids):
            idx  = np.where(class_ids == cls)[0]
            kept = cv2.dnn.NMSBoxes(
                boxes[idx].tolist(), confs[idx].tolist(),
                self._conf_thr, self._iou_thr,
            )
            if len(kept) == 0:
                continue
            kept = np.array(kept).flatten()
            for i in kept:
                ii = idx[i]
                result.append({
                    "class_id":  int(class_ids[ii]),
                    "conf":      float(confs[ii]),
                    "bbox_xyxy": (int(x1[ii]), int(y1[ii]), int(x2[ii]), int(y2[ii])),
                })
        return result


# ─── ReID gallery ─────────────────────────────────────────────────────────────

@dataclass
class _GalleryEntry:
    hist:    np.ndarray
    last_ts: float


class _ReidGallery:
    """Galería de histogramas HSV por track_id con TTL de expiración."""

    def __init__(self, ttl: float = REID_GALLERY_TTL) -> None:
        self._store: dict[int, _GalleryEntry] = {}
        self._ttl   = ttl

    def update(self, track_id: int, hist: np.ndarray) -> None:
        self._store[track_id] = _GalleryEntry(hist=hist, last_ts=time.time())

    def best_match(self, hist: np.ndarray, exclude: set[int]) -> Optional[int]:
        """Devuelve el track_id más similar con cosine ≥ REID_THRESHOLD."""
        now     = time.time()
        expired = [tid for tid, e in self._store.items() if now - e.last_ts > self._ttl]
        for tid in expired:
            del self._store[tid]

        best_id, best_sim = None, REID_THRESHOLD - 1e-6
        norm_q = np.linalg.norm(hist) + 1e-8
        for tid, entry in self._store.items():
            if tid in exclude:
                continue
            sim = float(np.dot(hist, entry.hist) / (norm_q * (np.linalg.norm(entry.hist) + 1e-8)))
            if sim > best_sim:
                best_sim, best_id = sim, tid
        return best_id

    def clear(self) -> None:
        self._store.clear()


# ─── ByteTrack + ReID ─────────────────────────────────────────────────────────

class ByteTrackReID:
    """
    ByteTrack (supervision) + galería ReID HSV cosine similarity.
    Mismo algoritmo que el traffic counter de Orange Pi, adaptado para ciclismo.
    """

    def __init__(self, reid_threshold: float = REID_THRESHOLD) -> None:
        from supervision.tracker.byte_tracker.core import ByteTrack as _ByteTrack
        import supervision as sv

        self._sv        = sv
        self._bt        = _ByteTrack(
            track_activation_threshold = 0.25,
            lost_track_buffer          = 60,
            minimum_matching_threshold = 0.8,
            frame_rate                 = 30,
        )
        self._gallery:    dict[int, _GalleryEntry] = {}   # display_id → entry
        self._hist_cache: dict[int, np.ndarray]    = {}   # bt_id → hist
        self._id_remap:   dict[int, int]           = {}   # bt_id → display_id
        self._active_bt_ids: set[int]              = set()
        self._last_date  = _date.today()
        self.reid_enabled = True
        self._reid_thr    = reid_threshold
        logger.info("ByteTrackReID listo (cosine threshold=%.2f)", reid_threshold)

    def update(self, detections: list[dict], frame: np.ndarray) -> list[Track]:
        """Actualiza con detecciones del frame; retorna tracks activos."""
        # Medianoche: limpiar galería
        today = _date.today()
        if today != self._last_date:
            self._gallery.clear()
            self._last_date = today
            logger.info("Galería ReID reiniciada (nuevo día)")

        if detections:
            bboxes  = np.array([d["bbox_xyxy"] for d in detections], dtype=np.float32)
            confs   = np.array([d["conf"]      for d in detections], dtype=np.float32)
            cls_ids = np.array([d["class_id"]  for d in detections], dtype=int)
        else:
            bboxes  = np.empty((0, 4), dtype=np.float32)
            confs   = np.empty((0,),   dtype=np.float32)
            cls_ids = np.empty((0,),   dtype=int)

        sv_dets = self._sv.Detections(xyxy=bboxes, confidence=confs, class_id=cls_ids)
        tracked = self._bt.update_with_detections(sv_dets)

        if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
            self._flush_to_gallery(self._active_bt_ids)
            self._active_bt_ids = set()
            return []

        current_bt_ids = {int(tid) for tid in tracked.tracker_id}

        # Tracks perdidos → guardar en galería
        lost_ids = self._active_bt_ids - current_bt_ids
        self._flush_to_gallery(lost_ids)
        self._cleanup_gallery()

        # Tracks nuevos → intentar ReID
        new_bt_ids = current_bt_ids - self._active_bt_ids
        for bt_id in new_bt_ids:
            idx  = list(int(t) for t in tracked.tracker_id).index(bt_id)
            bbox = tuple(int(v) for v in tracked.xyxy[idx])
            if frame is not None and self.reid_enabled:
                hist = _compute_hsv_hist(frame, *bbox)
                if hist is not None:
                    self._hist_cache[bt_id] = hist
                    match_id = self._gallery_match(hist)
                    if match_id is not None:
                        self._id_remap[bt_id] = match_id
                        logger.info("ReID: bt=%d → id=%d restaurado", bt_id, match_id)

        self._active_bt_ids = current_bt_ids

        tracks = []
        for i in range(len(tracked)):
            bt_id = int(tracked.tracker_id[i])
            tid   = self._id_remap.get(bt_id, bt_id)
            x1, y1, x2, y2 = map(int, tracked.xyxy[i])
            y_centro = (y1 + y2) / 2.0
            conf_val = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

            # Actualizar histograma del track en caché
            if frame is not None and self.reid_enabled:
                hist = _compute_hsv_hist(frame, x1, y1, x2, y2)
                if hist is not None:
                    self._hist_cache[bt_id] = hist

            tracks.append(Track(
                track_id  = tid,
                bbox_xyxy = (x1, y1, x2, y2),
                y_centro  = y_centro,
                con_casco = False,
                conf      = conf_val,
            ))

        return tracks

    def _flush_to_gallery(self, bt_ids: set[int]) -> None:
        now = time.time()
        for bt_id in bt_ids:
            hist = self._hist_cache.pop(bt_id, None)
            if hist is None:
                continue
            display_id = self._id_remap.pop(bt_id, bt_id)
            self._gallery[display_id] = _GalleryEntry(hist=hist, last_ts=now)

    def _cleanup_gallery(self) -> None:
        now     = time.time()
        expired = [did for did, e in self._gallery.items()
                   if now - e.last_ts > REID_GALLERY_TTL]
        for did in expired:
            del self._gallery[did]

    def _gallery_match(self, hist: np.ndarray) -> Optional[int]:
        active_display = set(self._id_remap.values())
        best_id, best_sim = None, self._reid_thr - 1e-6
        norm_q = np.linalg.norm(hist) + 1e-8
        for did, entry in self._gallery.items():
            if did in active_display:
                continue
            sim = float(np.dot(hist, entry.hist) /
                        (norm_q * (np.linalg.norm(entry.hist) + 1e-8)))
            if sim > best_sim:
                best_sim, best_id = sim, did
        return best_id

    def active_ids(self) -> set[int]:
        return set(self._active_bt_ids)


def _compute_hsv_hist(frame: np.ndarray, x1, y1, x2, y2) -> Optional[np.ndarray]:
    """Histograma HSV 16×16×16 normalizado del crop."""
    try:
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h   = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 16],
                           [0, 180, 0, 256, 0, 256])
        cv2.normalize(h, h)
        return h.flatten()
    except Exception:
        return None


# ─── CUDAEngine — interfaz pública ────────────────────────────────────────────

class CUDAEngine:
    """
    Equivalente al NPUEngine de Orange Pi, corriendo en PC Linux x86_64
    con CUDA (GTX 1650 o superior).

    Combina YOLODetector (onnxruntime-gpu) + ByteTrackReID.
    """

    def __init__(
        self,
        model_path: str,
        conf_thr:   float = 0.35,
        iou_thr:    float = 0.45,
    ) -> None:
        self._detector = YOLODetector(
            model_path,
            conf_thr       = conf_thr,
            iou_thr        = iou_thr,
            target_classes = [_PERSON_CLASS],  # solo personas (ciclistas)
        )
        self._tracker = ByteTrackReID()

    @property
    def provider(self) -> str:
        return self._detector.provider

    def run(self, frame: np.ndarray) -> list[Track]:
        """Detección + tracking en un frame. Devuelve tracks activos."""
        dets = self._detector.detect(frame)
        return self._tracker.update(dets, frame)

    def active_track_ids(self) -> set[int]:
        return self._tracker.active_ids()
