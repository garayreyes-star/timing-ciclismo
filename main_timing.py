"""
main_timing.py — Sistema de cronometraje de ciclistas para PC Linux x86_64.

Port del sistema Orange Pi (NPU Zhouyi + Mali OpenCL) a PC con CUDA GTX 1650:

  Captura VideoCapture (USB / RTSP / archivo)
      └─► Overlay CPU: linea de meta + bboxes + HUD
  CUDA (onnxruntime-gpu)
      └─► YOLO → ByteTrack+ReID → DetectorCruce
  Al cruzar la linea:
      ├─► Terminal: banner con tiempo de cruce (nanosegundos)
      ├─► CropWorker: recorte JPEG del ciclista guardado en data/captures/
      ├─► AsyncDB: registro en SQLite
      └─► PopupCruce: ventana Tkinter (foto + entrada de dorsal + podio)
  JudgeServer: panel web en http://0.0.0.0:8080 para tablet del juez

Uso:
    python3 main_timing.py
    python3 main_timing.py --camera 0 --model models/yolov8n.onnx --display
    python3 main_timing.py --camera rtsp://192.168.1.10/stream --no-display
    python3 main_timing.py --linea 540 --conf 0.40
"""

from __future__ import annotations

import os
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts*=false;qt.qpa.gl*=false")

import argparse
import logging
import queue
import signal
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from detector            import CUDAEngine, Track
from utils.crossing      import DetectorCruce, LINEA_META_Y, TOLERANCIA
from utils.db_manager    import DBManager
from utils.judge_server  import JudgeServer
from utils.crop_worker   import CropJob, DBRecord, CropWorker
from utils.popup_cruce   import PopupCruce
from utils.config_panel  import ConfigPanel

# ─── Ventana de validación OpenCV ─────────────────────────────────────────────
_WIN_VALIDATION  = "Juez — Validacion de Cruce"
_VALIDATION_SECS = 10          # segundos que se muestra la ventana

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)-7s [%(threadName)s] %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── ANSI ─────────────────────────────────────────────────────────────────────

_AMARILLO = "\033[1;33m"
_CIAN     = "\033[1;36m"
_VERDE    = "\033[1;32m"
_ROJO     = "\033[1;31m"
_RESET    = "\033[0m"
_BOLD     = "\033[1m"

# ─── Configuracion por defecto ─────────────────────────────────────────────────

CAMERA_SRC   = 0
MODEL_PATH   = "models/yolov8n.onnx"
DB_PATH      = "data/carreras.db"
CAPTURES_DIR = "data/captures"
TARGET_FPS   = 30
SHOW_DISPLAY = True

FRAME_BUDGET_NS = 1_000_000_000 // TARGET_FPS

# ─── Colores overlay (BGR) ────────────────────────────────────────────────────

_CL_META    = (0,   0, 255)
_CL_BANDA   = (0, 165, 255)
_CL_BBOX    = (0, 220,  80)
_CL_CRUZADO = (255, 180,   0)
_CL_TEXTO   = (255, 255, 255)


# ─── Mensajes inter-hilo ──────────────────────────────────────────────────────

@dataclass
class _RawFrame:
    frame: np.ndarray
    ts_ns: int


@dataclass
class _DetFrame:
    frame:  np.ndarray
    tracks: list[Track]
    ts_ns:  int


# ─── Log de cruce en terminal ─────────────────────────────────────────────────

def _log_cruce(track_id: int, ts_ns: int, dorsal: str = "N/A") -> None:
    dt         = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).astimezone()
    tiempo_str = dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"
    msg_id     = f"Ciclista {track_id:>4} cruzo la meta!"
    msg_tiempo = f"Tiempo: {tiempo_str}"
    msg_dorsal = f"Dorsal: {dorsal}" if dorsal != "N/A" else ""
    sep        = "=" * 48

    print(
        f"\n  +{sep}+\n"
        f"  |  {_AMARILLO}{msg_id:<46}{_RESET}|\n"
        f"  |  {_CIAN}{msg_tiempo:<24}{_RESET}"
        f"{_VERDE}{msg_dorsal:<22}{_RESET}|\n"
        f"  +{sep}+\n",
        flush=True,
    )
    logger.info("%s  %s  %s", msg_id, msg_tiempo, msg_dorsal)


# ─── Overlay CPU ──────────────────────────────────────────────────────────────

def _draw_overlay(
    frame:           np.ndarray,
    tracks:          list[Track],
    cruzados:        set[int],
    fps_real:        float,
    inf_ms:          float,
    provider:        str,
    linea_y:         int,
    timing_started:  bool = False,
    start_time_ns:   Optional[int] = None,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # ── Línea de meta (limpia, sin adornos) ──────────────────────────────────
    cv2.line(out, (0, linea_y - TOLERANCIA), (w, linea_y - TOLERANCIA), _CL_BANDA, 1)
    cv2.line(out, (0, linea_y + TOLERANCIA), (w, linea_y + TOLERANCIA), _CL_BANDA, 1)
    cv2.line(out, (0, linea_y),              (w, linea_y),              _CL_META,  2)
    cv2.putText(out, "META", (8, linea_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, _CL_META, 2, cv2.LINE_AA)

    # ── Bboxes + punto grueso de detección de ID sobre el atleta ─────────────
    for t in tracks:
        x1, y1, x2, y2 = t.bbox_xyxy
        color = _CL_CRUZADO if t.track_id in cruzados else _CL_BBOX
        casco = "Si" if t.con_casco else "No"
        label = f"#{t.track_id}  casco:{casco}"

        # Bbox principal
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(out, (x1, y1 - 20), (x1 + len(label) * 9, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Punto grueso = zona de lectura del dorsal (30%–82% de altura del bbox)
        bh = y2 - y1
        oy1 = y1 + int(bh * 0.30)
        oy2 = y1 + int(bh * 0.82)
        ocx = (x1 + x2) // 2
        ocy = (oy1 + oy2) // 2
        # Rectángulo de la zona OCR
        cv2.rectangle(out, (x1 + 4, oy1), (x2 - 4, oy2), (0, 220, 255), 2)
        # Punto grueso central de detección
        cv2.circle(out, (ocx, ocy), 10, (0, 180, 255), -1)
        cv2.circle(out, (ocx, ocy), 10, (255, 255, 255),  2)
        cv2.circle(out, (ocx, ocy),  4, (0,   0,   0), -1)
        # Etiqueta "ID"
        cv2.putText(out, "ID", (x1 + 6, oy1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

    # HUD
    prov_short = "CUDA" if "CUDA" in provider else "CPU"
    hud = [
        f"FPS:    {fps_real:5.1f} / {TARGET_FPS}",
        f"Inf:    {inf_ms:.1f} ms",
        f"Prov:   {prov_short}",
        f"Tracks: {len(tracks)}",
    ]
    for i, line in enumerate(hud):
        cv2.putText(out, line, (w - 200, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, _CL_TEXTO, 1, cv2.LINE_AA)

    # Estado de carrera (esquina inferior izquierda)
    if timing_started and start_time_ns is not None:
        elapsed_ms = (time.time_ns() - start_time_ns) // 1_000_000
        total_s    = elapsed_ms // 1000
        minutos    = total_s // 60
        segundos   = total_s % 60
        ms_resto   = elapsed_ms % 1000
        estado_txt = f"CARRERA  {minutos:02d}:{segundos:02d}.{ms_resto:03d}"
        estado_col = (0, 230, 80)   # verde
    else:
        estado_txt = "ESPERANDO INICIO"
        estado_col = (0, 80, 255)   # rojo-naranja

    (tw, th), _ = cv2.getTextSize(estado_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (6, h - th - 14), (tw + 14, h - 4), (0, 0, 0), -1)
    cv2.putText(out, estado_txt, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, estado_col, 2, cv2.LINE_AA)

    return out


def _draw_crossing_panel(frame: np.ndarray, cruce: dict) -> np.ndarray:
    """Miniatura del ultimo cruce en esquina inferior derecha."""
    if not cruce or not cruce.get("foto_path"):
        return frame
    try:
        foto = cv2.imread(cruce["foto_path"])
        if foto is None:
            return frame
    except Exception:
        return frame

    h, w     = frame.shape[:2]
    panel_h  = panel_w = 200
    margin   = 12
    x1 = w - panel_w - margin
    y1 = h - panel_h - margin - 50
    x2, y2 = x1 + panel_w, y1 + panel_h

    foto = cv2.resize(foto, (panel_w, panel_h))
    cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 50 + 3), (0, 255, 0), 3)
    cv2.rectangle(frame, (x1, y1), (x2, y2 + 50), (0, 0, 0), -1)
    frame[y1:y2, x1:x2] = foto
    cv2.rectangle(frame, (x1, y1 - 26), (x2, y1), (0, 200, 0), -1)
    cv2.putText(frame, f"  CRUCE #{cruce['track_id']}", (x1 + 4, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, cruce["tiempo_str"], (x1 + 8, y2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


# ─── Beep ─────────────────────────────────────────────────────────────────────

def _play_beep(freq: int = 880, duration_ms: int = 180, volume: float = 0.65) -> None:
    """Beep no bloqueante: tono sinusoidal via sounddevice, fallback a terminal."""
    def _beep() -> None:
        try:
            import sounddevice as sd
            sr  = 44100
            t   = np.linspace(0, duration_ms / 1000,
                              int(sr * duration_ms / 1000), endpoint=False)
            tone = (np.sin(2 * np.pi * freq * t) * volume * 32767).astype(np.int16)
            sd.play(tone, sr)
            sd.wait()
        except Exception:
            try:
                os.system("beep -f 880 -l 180 2>/dev/null || true")
            except Exception:
                print("\a", end="", flush=True)   # bell de terminal como último recurso

    threading.Thread(target=_beep, daemon=True).start()


# ─── Ventana de validación ────────────────────────────────────────────────────

def _make_validation_frame(
    crop:      np.ndarray,
    dorsal_ocr: Optional[str],
    track_id:   int,
    until_ns:   int,
) -> np.ndarray:
    """
    Construye el frame de la ventana de validación del juez.
    Muestra el recorte del ciclista + resultado OCR + barra de tiempo.
    """
    TARGET_W = 520
    h, w = crop.shape[:2]
    if w != TARGET_W:
        scale = TARGET_W / w
        crop  = cv2.resize(crop, (TARGET_W, int(h * scale)),
                           interpolation=cv2.INTER_LANCZOS4)
    h, w = crop.shape[:2]

    PANEL_H = 110
    canvas  = np.zeros((h + PANEL_H, w, 3), dtype=np.uint8)
    canvas[:h, :w] = crop

    # ── Barra de tiempo restante (top 7 px) ──────────────────────────────────
    remaining = max(0.0, (until_ns - time.time_ns()) / 1e9)
    ratio     = min(1.0, remaining / _VALIDATION_SECS)
    bar_w     = int(w * ratio)
    bar_col   = (0, 200, 80) if ratio > 0.35 else (0, 120, 255)
    cv2.rectangle(canvas, (0, 0), (w, 7), (40, 40, 40), -1)
    cv2.rectangle(canvas, (0, 0), (bar_w, 7), bar_col, -1)

    # ── Header sobre la foto ──────────────────────────────────────────────────
    cv2.rectangle(canvas, (0, 7), (w, 38), (10, 10, 10), -1)
    cv2.putText(canvas, f"CRUCE  Track #{track_id}",
                (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Panel inferior ────────────────────────────────────────────────────────
    py = h
    cv2.rectangle(canvas, (0, py), (w, py + PANEL_H), (18, 18, 18), -1)
    cv2.line(canvas, (0, py), (w, py), (60, 60, 60), 1)

    if dorsal_ocr:
        # OCR encontró un número → mostrarlo grande
        cv2.putText(canvas, "OCR detectado:", (10, py + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"# {dorsal_ocr}",
                    (10, py + 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 255, 120), 3, cv2.LINE_AA)

        # Superponer número también sobre la foto (torso)
        overlay_y = max(7, h - 60)
        cv2.rectangle(canvas, (0, overlay_y), (w, h), (0, 0, 0), -1)
        cv2.putText(canvas, f"# {dorsal_ocr}",
                    (8, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 120), 3, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "OCR: sin deteccion", (10, py + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 140, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Ingresar dorsal en popup", (10, py + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1, cv2.LINE_AA)

    cv2.putText(canvas, "ESC = cerrar ventana",
                (10, py + PANEL_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)
    return canvas


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class TimingPipeline:
    """
    Orquesta 4 hilos del sistema de cronometraje.

    Hilo principal  → Captura + overlay CPU + imshow
    Thread CUDA     → Inferencia YOLO + ByteTrack + ReID
    Thread Crossing → DetectorCruce → log + CropJob
    Thread CropW    → CropWorker (JPEG a disco)
    Thread DB       → AsyncDBWriter (SQLite)
    """

    def __init__(
        self,
        camera_src:   int | str = CAMERA_SRC,
        model_path:   str       = MODEL_PATH,
        db_path:      str       = DB_PATH,
        captures_dir: str       = CAPTURES_DIR,
        show_display: bool      = SHOW_DISPLAY,
        linea_y:      int       = LINEA_META_Y,
        conf_thr:     float     = 0.35,
    ) -> None:
        self._camera_src   = camera_src
        self._show_display = show_display
        self._linea_y      = linea_y

        # Colas inter-hilo
        self._frame_deque: deque[_RawFrame] = deque(maxlen=1)
        self._detect_q:    queue.Queue      = queue.Queue(maxsize=2)
        self._crop_q:      queue.Queue      = queue.Queue(maxsize=16)
        self._db_q:        queue.Queue      = queue.Queue(maxsize=64)

        # Ajustar linea de meta si difiere del default
        if linea_y != LINEA_META_Y:
            import utils.crossing as _c
            _c.LINEA_META_Y = linea_y

        # Componentes
        self._cuda     = CUDAEngine(model_path, conf_thr=conf_thr)
        self._db       = DBManager(db_path)
        self._detector = DetectorCruce(db=None)
        self._crop_w   = CropWorker(self._crop_q, self._db_q, captures_dir)
        self._popup    = PopupCruce(
            self._db,
            on_inicio_carrera=self.iniciar_carrera,
            on_fin_carrera=self.finalizar_carrera,
        )

        self._stop = threading.Event()

        # Estado de inicio de carrera
        self._timing_started: bool          = False
        self._start_time_ns:  Optional[int] = None
        self._timing_lock     = threading.Lock()

        # Dedup: cada track_id solo genera UNA foto y UN registro por sesión
        self._cruzados_permanentes: set[int] = set()
        self._cruzados_perm_lock    = threading.Lock()

        # Estado compartido para overlay
        self._current_tracks: list[Track] = []
        self._cruzados_ids:   set[int]    = set()
        self._tracks_lock     = threading.Lock()

        self._last_cruce:      dict = {}
        self._last_cruce_lock  = threading.Lock()

        # Metricas
        self._fps_real  = 0.0
        self._inf_ms    = 0.0
        self._n_frames  = 0
        self._n_cruces  = 0

        # Ventana de validación OpenCV
        self._val_img:      Optional[np.ndarray] = None
        self._val_until_ns: int                  = 0
        self._val_track_id: int                  = 0
        self._val_ocr:      Optional[str]        = None
        self._val_ocr_prev: Optional[str]        = None   # para detectar cambios
        self._val_cache:    Optional[np.ndarray] = None   # frame precalculado
        self._val_lock      = threading.Lock()
        self._val_win_open  = False

        # Cola para OCR asíncrono: (id_reg, crop_np)
        self._ocr_q: queue.Queue = queue.Queue(maxsize=8)

        # Parámetros ajustables en tiempo real
        self._beep_freq: int = 880
        self._beep_ms:   int = 180
        self._cam_pending: dict          = {}
        self._cam_lock:    threading.Lock = threading.Lock()

        # Panel de configuración en tiempo real (hilo Tkinter daemon)
        self._config = ConfigPanel(on_change=self._on_config_change)
        # Aplicar valores iniciales del panel (beep, reid, cooldown)
        self._beep_freq = int(self._config.get("beep_freq"))
        self._beep_ms   = int(self._config.get("beep_ms"))
        import utils.crossing as _cx
        _cx.COOLDOWN_NS           = int(self._config.get("cooldown_s") * 1_000_000_000)
        self._cuda._tracker._reid_thr = self._config.get("reid_thr")

    # ──────────────────────────────────────────────────────────────────────────

    def _on_config_change(self, key: str, value: float) -> None:
        """Aplica en tiempo real los cambios del ConfigPanel al pipeline."""
        import utils.crossing as _cx
        if key == "conf_thr":
            self._cuda._detector._conf_thr = value
        elif key == "iou_thr":
            self._cuda._detector._iou_thr = value
        elif key == "linea_y":
            self._linea_y    = int(value)
            _cx.LINEA_META_Y = int(value)
        elif key == "reid_thr":
            self._cuda._tracker._reid_thr = value
        elif key == "cooldown_s":
            _cx.COOLDOWN_NS = int(value * 1_000_000_000)
        elif key == "beep_freq":
            self._beep_freq = int(value)
        elif key == "beep_ms":
            self._beep_ms = int(value)
        elif key.startswith("cam_"):
            prop_map = {
                "cam_brillo":     cv2.CAP_PROP_BRIGHTNESS,
                "cam_contraste":  cv2.CAP_PROP_CONTRAST,
                "cam_saturacion": cv2.CAP_PROP_SATURATION,
                "cam_exposicion": cv2.CAP_PROP_EXPOSURE,
            }
            if key in prop_map:
                with self._cam_lock:
                    self._cam_pending[prop_map[key]] = value

    def iniciar_carrera(self) -> None:
        """Activa el marcaje. Llamado desde el botón de PopupCruce."""
        with self._timing_lock:
            if self._timing_started:
                return
            self._start_time_ns  = time.time_ns()
            self._timing_started = True
        self._db.guardar_inicio_carrera(self._start_time_ns)
        logger.info(
            "%s[CARRERA]%s  Inicio de marcaje — ts_ns=%d",
            _BOLD, _RESET, self._start_time_ns,
        )

    def finalizar_carrera(self) -> str:
        """
        Detiene el marcaje, guarda el fin en BD y exporta resultados a CSV.
        Retorna la ruta del CSV generado.
        Llamado desde el botón FINALIZAR CARRERA del popup.
        """
        fin_ns = time.time_ns()
        with self._timing_lock:
            self._timing_started = False
        self._db.guardar_fin_carrera(fin_ns)

        # Nombre de archivo con fecha y hora
        fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"data/resultados_{fecha}.csv"
        n_filas = self._db.exportar_csv(csv_path)

        logger.info(
            "%s[FIN CARRERA]%s  ts_ns=%d  CSV=%s  (%d registros)",
            _BOLD, _RESET, fin_ns, csv_path, n_filas,
        )
        return csv_path

    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        cap = self._open_camera()

        workers = [
            threading.Thread(target=self._inference_loop, name="CUDA-YOLO",   daemon=True),
            threading.Thread(target=self._crossing_loop,  name="CrossingDet", daemon=True),
            self._crop_w,
            threading.Thread(target=self._db_loop,        name="AsyncDB",     daemon=True),
            threading.Thread(target=self._ocr_loop,       name="OCR-Async",   daemon=True),
        ]
        for w in workers:
            w.start()

        judge = JudgeServer(self._db)
        judge.start()

        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            ip = "localhost"
        logger.info("%s[JUEZ]%s  Panel en http://%s:8080", _BOLD, _RESET, ip)
        logger.info(
            "%s[INICIO]%s  camara=%s  modelo=%s  provider=%s  linea_y=%d",
            _BOLD, _RESET, self._camera_src, MODEL_PATH,
            self._cuda.provider, self._linea_y,
        )

        try:
            self._capture_loop(cap)
        finally:
            self._stop.set()
            self._crop_w.stop()
            cap.release()
            cv2.destroyAllWindows()
            self._db.close()
            logger.info(
                "%s[FIN]%s  frames=%d  cruces=%d",
                _BOLD, _RESET, self._n_frames, self._n_cruces,
            )

    # ── Captura + overlay ─────────────────────────────────────────────────────

    def _capture_loop(self, cap: cv2.VideoCapture) -> None:
        deadline_ns  = time.perf_counter_ns()
        fps_ts_start = time.perf_counter()
        fps_count    = 0

        win_name = "Cronometraje — CUDA YOLO + ByteTrack"
        if self._show_display:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 1280, 720)

        while not self._stop.is_set():
            # Aplicar ajustes de cámara pendientes del ConfigPanel
            with self._cam_lock:
                if self._cam_pending and isinstance(self._camera_src, int):
                    for prop, val in self._cam_pending.items():
                        cap.set(prop, val)
                    self._cam_pending.clear()

            ok, frame = cap.read()
            if not ok:
                logger.warning("cap.read() fallo — reintentando...")
                time.sleep(0.01)
                continue

            ts_ns = time.time_ns()
            self._frame_deque.append(_RawFrame(frame, ts_ns))
            self._n_frames += 1
            fps_count      += 1

            if fps_count == 30:
                elapsed          = time.perf_counter() - fps_ts_start
                self._fps_real   = 30 / elapsed if elapsed > 0 else 0
                fps_ts_start     = time.perf_counter()
                fps_count        = 0

            if self._show_display:
                with self._tracks_lock:
                    tracks_snap   = list(self._current_tracks)
                    cruzados_snap = set(self._cruzados_ids)
                with self._last_cruce_lock:
                    last_snap = dict(self._last_cruce)

                with self._timing_lock:
                    t_started  = self._timing_started
                    t_start_ns = self._start_time_ns

                rendered = _draw_overlay(
                    frame, tracks_snap, cruzados_snap,
                    self._fps_real, self._inf_ms,
                    self._cuda.provider, self._linea_y,
                    timing_started=t_started,
                    start_time_ns=t_start_ns,
                )
                rendered = _draw_crossing_panel(rendered, last_snap)
                cv2.imshow(win_name, rendered)

                # ── Ventana de validación del juez ────────────────────────────
                with self._val_lock:
                    now_ns    = time.time_ns()
                    val_active = (
                        self._val_img is not None
                        and now_ns < self._val_until_ns
                    )
                    if val_active:
                        _v_img      = self._val_img
                        _v_until    = self._val_until_ns
                        _v_track    = self._val_track_id
                        _v_ocr      = self._val_ocr

                if val_active:
                    # Reconstruir solo cuando cambia el OCR, no cada frame
                    if self._val_ocr_prev != _v_ocr or self._val_cache is None:
                        self._val_cache    = _make_validation_frame(
                            _v_img.copy(), _v_ocr, _v_track, _v_until
                        )
                        self._val_ocr_prev = _v_ocr
                    # Agregar barra de countdown (liviana, solo 7px)
                    v_display = self._val_cache.copy()
                    _remaining = max(0.0, (_v_until - time.time_ns()) / 1e9)
                    _bw = int(v_display.shape[1] * _remaining / _VALIDATION_SECS)
                    _bc = (0, 200, 80) if _remaining > 3.5 else (0, 120, 255)
                    cv2.rectangle(v_display, (0, 0), (v_display.shape[1], 7), (40,40,40), -1)
                    cv2.rectangle(v_display, (0, 0), (_bw, 7), _bc, -1)
                    if not self._val_win_open:
                        cv2.namedWindow(_WIN_VALIDATION, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(_WIN_VALIDATION, 520, 640)
                        self._val_win_open = True
                    cv2.imshow(_WIN_VALIDATION, v_display)
                elif self._val_win_open:
                    cv2.destroyWindow(_WIN_VALIDATION)
                    self._val_win_open = False

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == 27 and self._val_win_open:   # ESC cierra validación
                    with self._val_lock:
                        self._val_until_ns = 0

            # Rate-limiter nanosegundos
            deadline_ns += FRAME_BUDGET_NS
            slack_ns     = deadline_ns - time.perf_counter_ns()
            if slack_ns > 500_000:
                time.sleep(slack_ns * 1e-9)
            elif slack_ns < -FRAME_BUDGET_NS:
                deadline_ns = time.perf_counter_ns()

    # ── Thread CUDA-YOLO ──────────────────────────────────────────────────────

    def _inference_loop(self) -> None:
        while not self._stop.is_set():
            if not self._frame_deque:
                time.sleep(0.001)
                continue

            raw = self._frame_deque[0]

            t0     = time.perf_counter_ns()
            tracks = self._cuda.run(raw.frame)
            self._inf_ms = (time.perf_counter_ns() - t0) / 1e6

            with self._tracks_lock:
                self._current_tracks = tracks

            df = _DetFrame(frame=raw.frame, tracks=tracks, ts_ns=raw.ts_ns)
            try:
                self._detect_q.put_nowait(df)
            except queue.Full:
                try:
                    self._detect_q.get_nowait()
                except queue.Empty:
                    pass
                self._detect_q.put_nowait(df)

            # Limpiar tracks extintos del detector de cruce
            active = self._cuda.active_track_ids()
            for tid in list(self._detector._tracks):
                if tid not in active:
                    self._detector.limpiar_track(tid)

    # ── Thread Crossing ───────────────────────────────────────────────────────

    def _crossing_loop(self) -> None:
        while not self._stop.is_set():
            try:
                df: _DetFrame = self._detect_q.get(timeout=0.1)
            except queue.Empty:
                continue

            for track in df.tracks:
                cruzo = self._detector.detectar_cruce(
                    track_id  = track.track_id,
                    y_centro  = track.y_centro,
                    con_casco = track.con_casco,
                )
                if not cruzo:
                    continue

                # Solo registrar si la carrera fue iniciada
                with self._timing_lock:
                    if not self._timing_started:
                        continue
                    start_ns = self._start_time_ns

                # Dedup: cada track_id se registra UNA SOLA VEZ por sesión
                with self._cruzados_perm_lock:
                    if track.track_id in self._cruzados_permanentes:
                        continue
                    self._cruzados_permanentes.add(track.track_id)

                ts_ns = self._detector._tracks[track.track_id].ultimo_cruce_ns
                tiempo_carrera_ms = (ts_ns - start_ns) // 1_000_000

                _play_beep(self._beep_freq, self._beep_ms)   # beep no bloqueante
                _log_cruce(track.track_id, ts_ns)
                self._n_cruces += 1

                with self._tracks_lock:
                    self._cruzados_ids.add(track.track_id)

                tid_local = track.track_id
                threading.Timer(
                    3.0,
                    lambda t=tid_local: self._cruzados_ids.discard(t),
                ).start()

                self._crop_q.put(CropJob(
                    frame             = df.frame,
                    track_id          = track.track_id,
                    bbox_xyxy         = track.bbox_xyxy,
                    ts_ns             = ts_ns,
                    con_casco         = track.con_casco,
                    tiempo_carrera_ms = tiempo_carrera_ms,
                ))

    # ── Thread DB ─────────────────────────────────────────────────────────────

    def _db_loop(self) -> None:
        while not self._stop.is_set():
            try:
                rec: DBRecord = self._db_q.get(timeout=0.5)
            except queue.Empty:
                continue

            tiempo_cruce = datetime.fromtimestamp(rec.ts_ns / 1e9, tz=timezone.utc)
            id_reg = self._db.registrar_cruce(
                track_id          = rec.track_id,
                con_casco         = rec.con_casco,
                dorsal            = rec.dorsal,
                foto_meta_path    = rec.foto_meta_path,
                tiempo_cruce      = tiempo_cruce,
                tiempo_carrera_ms = rec.tiempo_carrera_ms,
            )

            dt         = tiempo_cruce.astimezone()
            tiempo_str = dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"
            with self._last_cruce_lock:
                self._last_cruce = {
                    "track_id":        rec.track_id,
                    "ts_ns":           rec.ts_ns,
                    "foto_path":       rec.foto_meta_path,
                    "tiempo_str":      tiempo_str,
                    "con_casco":       rec.con_casco,
                    "tiempo_carrera_ms": rec.tiempo_carrera_ms,
                }

            if id_reg is not None:
                # Mostrar popup inmediatamente (sin esperar OCR)
                self._popup.nuevo_cruce(
                    id_registro       = id_reg,
                    track_id          = rec.track_id,
                    ts_ns             = rec.ts_ns,
                    foto_path         = rec.foto_meta_path,
                    con_casco         = rec.con_casco,
                    tiempo_carrera_ms = rec.tiempo_carrera_ms,
                    dorsal_ocr        = None,   # OCR llega después de forma asíncrona
                )

                # Ventana de validación OpenCV: mostrar foto de inmediato
                _crop_disp = rec.crop_np if rec.crop_np is not None else (
                    cv2.imread(rec.foto_meta_path) if rec.foto_meta_path else None
                )
                if _crop_disp is not None:
                    with self._val_lock:
                        self._val_img      = _crop_disp
                        self._val_until_ns = time.time_ns() + int(_VALIDATION_SECS * 1e9)
                        self._val_track_id = rec.track_id
                        self._val_ocr      = None   # se actualiza cuando OCR termina

                # Encolar para OCR asíncrono
                if rec.crop_np is not None:
                    try:
                        self._ocr_q.put_nowait((id_reg, rec.track_id, rec.crop_np))
                    except queue.Full:
                        pass

    # ── Thread OCR asíncrono ──────────────────────────────────────────────────

    def _ocr_loop(self) -> None:
        """
        Corre Tesseract en segundo plano sin bloquear el pipeline principal.
        Cuando detecta un dorsal, actualiza el popup y la ventana de validación.
        """
        from utils.ocr_dorsal import detectar_dorsal
        while not self._stop.is_set():
            try:
                id_reg, track_id, crop_np = self._ocr_q.get(timeout=0.5)
            except queue.Empty:
                continue

            dorsal_ocr = detectar_dorsal(crop_np)
            if dorsal_ocr:
                logger.info("OCR track=%d → dorsal=%s", track_id, dorsal_ocr)

            # Actualizar ventana de validación si sigue siendo del mismo track
            with self._val_lock:
                if self._val_track_id == track_id:
                    self._val_ocr = dorsal_ocr

            # Notificar al popup para pre-rellenar el dorsal
            self._popup.actualizar_ocr(id_reg, dorsal_ocr)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _open_camera(self) -> cv2.VideoCapture:
        import platform
        if isinstance(self._camera_src, int):
            if platform.system() == "Windows":
                cap = cv2.VideoCapture(self._camera_src, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(self._camera_src, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(self._camera_src)

        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la camara: {self._camera_src!r}")

        if isinstance(self._camera_src, int):
            cap.set(cv2.CAP_PROP_FOURCC,      cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info("Camara: %dx%d @ %.0f FPS  src=%s", w, h, fps, self._camera_src)
        return cap


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cronometraje de ciclistas — CUDA YOLO + ByteTrack + ReID"
    )
    p.add_argument("--camera",   default=CAMERA_SRC,   help="Indice camara o URL RTSP")
    p.add_argument("--model",    default=MODEL_PATH,   help="Ruta al modelo ONNX")
    p.add_argument("--db",       default=DB_PATH,      help="Ruta a la base de datos SQLite")
    p.add_argument("--captures", default=CAPTURES_DIR, help="Dir para capturas JPEG")
    p.add_argument("--linea",    default=LINEA_META_Y, type=int, help="Y de la linea de meta")
    p.add_argument("--conf",     default=0.35,         type=float, help="Umbral de confianza YOLO")
    p.add_argument("--display",    action="store_true",  default=SHOW_DISPLAY)
    p.add_argument("--no-display", dest="display",       action="store_false")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    camera_src: int | str = args.camera
    try:
        camera_src = int(args.camera)
    except (ValueError, TypeError):
        pass

    pipeline = TimingPipeline(
        camera_src   = camera_src,
        model_path   = args.model,
        db_path      = args.db,
        captures_dir = args.captures,
        show_display = args.display,
        linea_y      = args.linea,
        conf_thr     = args.conf,
    )

    def _sigint(_sig, _frame):
        print(f"\n{_ROJO}  Interrupcion — cerrando...{_RESET}")
        pipeline._stop.set()

    signal.signal(signal.SIGINT, _sigint)

    try:
        pipeline.run()
    except RuntimeError as e:
        logger.error("%s%s%s", _ROJO, e, _RESET)
        sys.exit(1)


if __name__ == "__main__":
    main()
