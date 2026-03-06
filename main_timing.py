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
    frame:        np.ndarray,
    tracks:       list[Track],
    cruzados:     set[int],
    fps_real:     float,
    inf_ms:       float,
    provider:     str,
    linea_y:      int,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Linea de meta
    cv2.line(out, (0, linea_y - TOLERANCIA), (w, linea_y - TOLERANCIA), _CL_BANDA, 1)
    cv2.line(out, (0, linea_y + TOLERANCIA), (w, linea_y + TOLERANCIA), _CL_BANDA, 1)
    cv2.line(out, (0, linea_y),              (w, linea_y),              _CL_META,  2)
    cv2.putText(out, "META", (8, linea_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, _CL_META, 2, cv2.LINE_AA)

    # Bboxes de ciclistas
    for t in tracks:
        x1, y1, x2, y2 = t.bbox_xyxy
        color  = _CL_CRUZADO if t.track_id in cruzados else _CL_BBOX
        casco  = "Si" if t.con_casco else "No"
        label  = f"#{t.track_id}  casco:{casco}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(out, (x1, y1 - 20), (x1 + len(label) * 9, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

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
        self._popup    = PopupCruce(self._db)

        self._stop = threading.Event()

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

    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        cap = self._open_camera()

        workers = [
            threading.Thread(target=self._inference_loop, name="CUDA-YOLO",   daemon=True),
            threading.Thread(target=self._crossing_loop,  name="CrossingDet", daemon=True),
            self._crop_w,
            threading.Thread(target=self._db_loop,        name="AsyncDB",     daemon=True),
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
            cv2.resizeWindow(win_name, 960, 540)

        while not self._stop.is_set():
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

                rendered = _draw_overlay(
                    frame, tracks_snap, cruzados_snap,
                    self._fps_real, self._inf_ms,
                    self._cuda.provider, self._linea_y,
                )
                rendered = _draw_crossing_panel(rendered, last_snap)
                cv2.imshow(win_name, rendered)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

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
                if cruzo:
                    ts_ns = self._detector._tracks[track.track_id].ultimo_cruce_ns
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
                        frame     = df.frame,
                        track_id  = track.track_id,
                        bbox_xyxy = track.bbox_xyxy,
                        ts_ns     = ts_ns,
                        con_casco = track.con_casco,
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
                track_id       = rec.track_id,
                con_casco      = rec.con_casco,
                dorsal         = rec.dorsal,
                foto_meta_path = rec.foto_meta_path,
                tiempo_cruce   = tiempo_cruce,
            )

            dt         = tiempo_cruce.astimezone()
            tiempo_str = dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"
            with self._last_cruce_lock:
                self._last_cruce = {
                    "track_id":  rec.track_id,
                    "ts_ns":     rec.ts_ns,
                    "foto_path": rec.foto_meta_path,
                    "tiempo_str": tiempo_str,
                    "con_casco": rec.con_casco,
                }

            if id_reg is not None:
                self._popup.nuevo_cruce(
                    id_registro = id_reg,
                    track_id    = rec.track_id,
                    ts_ns       = rec.ts_ns,
                    foto_path   = rec.foto_meta_path,
                    con_casco   = rec.con_casco,
                )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _open_camera(self) -> cv2.VideoCapture:
        if isinstance(self._camera_src, int):
            cap = cv2.VideoCapture(self._camera_src, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(self._camera_src)

        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la camara: {self._camera_src!r}")

        if isinstance(self._camera_src, int):
            cap.set(cv2.CAP_PROP_FOURCC,      cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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
