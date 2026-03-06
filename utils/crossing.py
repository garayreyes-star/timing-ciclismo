"""
crossing.py — Detección de cruce de línea de meta con precisión de nanosegundos.

Algoritmo de zonas con histéresis (3 regiones):

    NORTE ────────────── y < LINEA_META_Y - TOLERANCIA
    ═════ BANDA DE META ═ [LINEA_META_Y-TOLERANCIA .. LINEA_META_Y+TOLERANCIA]
    SUR ──────────────── y > LINEA_META_Y + TOLERANCIA

Un cruce se dispara la primera vez que un track_id entra en BANDA
habiendo estado previamente en NORTE.  La lógica de histéresis evita
disparos falsos por jitter del tracker en la zona límite.

El cooldown por track_id (COOLDOWN_NS) protege contra dobles registros
cuando un corredor se detiene exactamente sobre la línea.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional

from utils.db_manager import DBManager

logger = logging.getLogger(__name__)

# ─── Parámetros de la línea de meta ──────────────────────────────────────────

LINEA_META_Y: int = 360       # Coordenada Y de la línea (píxeles).
                               # 360 = centro de frame 720p (1280x720).
                               # Ajustar según posición física de la cámara.
TOLERANCIA: int = 5            # Semiancho de la banda de cruce (píxeles).
COOLDOWN_NS: int = 3_000_000_000   # 3 s en nanosegundos — evita doble registro


# ─── Tipos internos ───────────────────────────────────────────────────────────

class Zona(Enum):
    NORTE = auto()   # el corredor se aproxima (y < LINEA_META_Y - TOLERANCIA)
    BANDA = auto()   # dentro de la zona de cruce
    SUR   = auto()   # ya cruzó (y > LINEA_META_Y + TOLERANCIA)


@dataclass
class _EstadoTrack:
    """Estado interno de un único track_id."""
    zona_previa:   Zona          # última zona fuera de BANDA (NORTE o SUR)
    en_banda:      bool = False  # True mientras y_centro está en la banda
    ultimo_cruce_ns: int = 0     # time.time_ns() del último cruce registrado


# ─── Detector ─────────────────────────────────────────────────────────────────

class DetectorCruce:
    """
    Mantiene el estado de cada track_id activo y decide cuándo insertar
    un cruce en la base de datos.

    Uso típico en el loop principal::

        detector = DetectorCruce(db)
        for frame in camara:
            tracks = bytetrack.update(frame)
            for t in tracks:
                cruzó = detector.detectar_cruce(
                    track_id=t.id,
                    y_centro=t.bbox_cy,
                    con_casco=t.con_casco,
                    dorsal=t.dorsal,
                    foto_meta_path=t.foto_path,
                )
    """

    def __init__(self, db: Optional[DBManager] = None) -> None:
        self._db = db
        self._tracks: dict[int, _EstadoTrack] = {}

    # ──────────────────────────────────────────────────────────────────────────

    def detectar_cruce(
        self,
        track_id:       int,
        y_centro:       float,
        con_casco:      bool,
        dorsal:         str = "N/A",
        foto_meta_path: Optional[str] = None,
    ) -> bool:
        """
        Evalúa si track_id ha cruzado la línea de meta en este frame.

        Args:
            track_id:       ID asignado por ByteTrack.
            y_centro:       Coordenada Y del centro del bounding box (píxeles).
            con_casco:      True si YOLO detectó casco en el mismo frame.
            dorsal:         Texto del OCR, o 'N/A'.
            foto_meta_path: Ruta a la captura guardada por la GPU Mali-G610.

        Returns:
            True si se registró un cruce nuevo, False en cualquier otro caso.
        """
        zona_actual = self._zona(y_centro)

        # ── Inicializar estado la primera vez que vemos este track ────────────
        if track_id not in self._tracks:
            self._tracks[track_id] = _EstadoTrack(zona_previa=zona_actual)
            return False

        estado = self._tracks[track_id]

        # ── Actualizar zona de referencia (solo fuera de BANDA) ───────────────
        if zona_actual != Zona.BANDA:
            cruce_detectado = (
                estado.zona_previa == Zona.NORTE       # venía del norte
                and zona_actual    == Zona.SUR         # ahora está al sur
                and estado.en_banda                    # pasó por la banda
            )
            estado.zona_previa = zona_actual
            estado.en_banda    = False

            if cruce_detectado:
                return self._registrar(estado, track_id, con_casco, dorsal, foto_meta_path)

        else:
            # El track acaba de entrar en la banda
            if not estado.en_banda:
                estado.en_banda = True

                # Cruce por entrada directa NORTE → BANDA
                if estado.zona_previa == Zona.NORTE:
                    return self._registrar(estado, track_id, con_casco, dorsal, foto_meta_path)

        return False

    # ──────────────────────────────────────────────────────────────────────────

    def limpiar_track(self, track_id: int) -> None:
        """Elimina el estado de un track que ya no está activo."""
        self._tracks.pop(track_id, None)

    def tracks_activos(self) -> int:
        return len(self._tracks)

    # ──────────────────────────────────────────────────────────────────────────
    #  Helpers privados
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _zona(y: float) -> Zona:
        if y < LINEA_META_Y - TOLERANCIA:
            return Zona.NORTE
        if y > LINEA_META_Y + TOLERANCIA:
            return Zona.SUR
        return Zona.BANDA

    def _registrar(
        self,
        estado:         _EstadoTrack,
        track_id:       int,
        con_casco:      bool,
        dorsal:         str,
        foto_meta_path: Optional[str],
    ) -> bool:
        """
        Emite el timestamp con time.time_ns() y persiste el cruce en la BD.
        Aplica cooldown para evitar dobles registros.
        """
        ahora_ns: int = time.time_ns()      # resolución de nanosegundos (CIX P1)

        if ahora_ns - estado.ultimo_cruce_ns < COOLDOWN_NS:
            logger.debug("Cruce de track %d ignorado por cooldown", track_id)
            return False

        estado.ultimo_cruce_ns = ahora_ns

        # Convertir ns → datetime con precisión de milisegundos para la BD
        tiempo_cruce = datetime.fromtimestamp(ahora_ns / 1e9, tz=timezone.utc)

        logger.info(
            "CRUCE  track=%-4d  dorsal=%-6s  casco=%s  t=%s  ns=%d",
            track_id, dorsal, "SI" if con_casco else "NO",
            tiempo_cruce.strftime("%H:%M:%S.%f")[:-3], ahora_ns,
        )

        # db es opcional: en el pipeline el registro lo hace AsyncDBWriter
        if self._db is not None:
            id_reg = self._db.registrar_cruce(
                track_id=track_id, con_casco=con_casco, dorsal=dorsal,
                foto_meta_path=foto_meta_path, tiempo_cruce=tiempo_cruce,
            )
            return id_reg is not None

        return True
