"""
config_panel.py — Panel de configuración en tiempo real del sistema de cronometraje.

Permite ajustar sin reiniciar:
  • YOLO: confianza y umbral IOU
  • Cámara: brillo, contraste, saturación, exposición
  • Línea de meta: posición Y
  • Tracking: umbral ReID, cooldown entre cruces
  • Beep: frecuencia y duración

Los cambios se guardan en data/config.json y se aplican al instante.
"""

from __future__ import annotations

import json
import logging
import threading
import tkinter as tk
from tkinter import font as tkfont
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "data" / "config.json"

# ─── Valores por defecto ──────────────────────────────────────────────────────

DEFAULTS: dict = {
    # YOLO
    "conf_thr":    0.35,
    "iou_thr":     0.45,
    # Cámara
    "cam_brillo":      0,
    "cam_contraste":  32,
    "cam_saturacion": 64,
    "cam_exposicion": -6,
    # Línea de meta
    "linea_y":    360,
    # Tracking / ReID
    "reid_thr":   0.65,
    "cooldown_s": 3.0,
    # Beep
    "beep_freq":  880,
    "beep_ms":    180,
}

# ─── Colores ──────────────────────────────────────────────────────────────────

C_BG     = "#0d0d0d"
C_PANEL  = "#1a1a1a"
C_GRUPO  = "#111111"
C_VERDE  = "#00e676"
C_AZUL   = "#29b6f6"
C_AMARILLO = "#ffd600"
C_ROJO   = "#ff1744"
C_TEXTO  = "#f0f0f0"
C_GRIS   = "#777777"
C_BORDE  = "#333333"


# ─── Clase principal ──────────────────────────────────────────────────────────

class ConfigPanel:
    """
    Ventana Tkinter de configuración en tiempo real.
    Corre en su propio hilo daemon.

    Callbacks registrados via ``on_change(key, value)`` se llaman
    desde el hilo Tkinter cada vez que un slider cambia.
    """

    def __init__(self, on_change: Optional[Callable[[str, float], None]] = None) -> None:
        self._on_change = on_change
        self._vars:     dict[str, tk.Variable] = {}
        self._cfg:      dict = self._load()
        threading.Thread(target=self._run, name="ConfigPanel", daemon=True).start()

    # ── Carga / guardado ──────────────────────────────────────────────────────

    def _load(self) -> dict:
        """Carga config.json; usa defaults para claves faltantes."""
        cfg = dict(DEFAULTS)
        if _CONFIG_PATH.exists():
            try:
                saved = json.loads(_CONFIG_PATH.read_text())
                cfg.update({k: saved[k] for k in DEFAULTS if k in saved})
                logger.info("Configuracion cargada: %s", _CONFIG_PATH)
            except Exception as e:
                logger.warning("Error leyendo config.json: %s — usando defaults", e)
        return cfg

    def _save(self) -> None:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(json.dumps(self._cfg, indent=2))
        logger.info("Configuracion guardada: %s", _CONFIG_PATH)

    def get(self, key: str) -> float:
        """Valor actual de una clave (float)."""
        return float(self._cfg.get(key, DEFAULTS[key]))

    # ── Hilo Tkinter ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        self._root = tk.Tk()
        self._root.title("Configuracion del Sistema")
        self._root.configure(bg=C_BG)
        self._root.resizable(False, False)
        self._build_ui()
        self._root.mainloop()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        r = self._root
        f_title = tkfont.Font(family="Arial", size=13, weight="bold")
        f_group = tkfont.Font(family="Arial", size=11, weight="bold")
        f_lbl   = tkfont.Font(family="Arial", size=10)
        f_val   = tkfont.Font(family="Courier", size=10, weight="bold")

        tk.Label(r, text="Configuracion del Sistema",
                 bg=C_BG, fg=C_VERDE, font=f_title).pack(pady=(12, 4))
        tk.Frame(r, bg=C_BORDE, height=1).pack(fill=tk.X, padx=10)

        container = tk.Frame(r, bg=C_BG)
        container.pack(fill=tk.BOTH, padx=14, pady=8)

        # ── columna izquierda y derecha ───────────────────────────────────────
        col_izq = tk.Frame(container, bg=C_BG)
        col_der = tk.Frame(container, bg=C_BG)
        col_izq.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        col_der.pack(side=tk.LEFT, fill=tk.Y)

        # ── YOLO ─────────────────────────────────────────────────────────────
        self._grupo(col_izq, "YOLO — Deteccion", f_group, [
            ("Confianza",    "conf_thr",    0.10, 0.90, 0.01, f_lbl, f_val, C_VERDE),
            ("IOU (NMS)",    "iou_thr",     0.20, 0.80, 0.01, f_lbl, f_val, C_VERDE),
        ])

        # ── Línea de meta ────────────────────────────────────────────────────
        self._grupo(col_izq, "Linea de Meta", f_group, [
            ("Posicion Y",   "linea_y",     0,   719,   1,    f_lbl, f_val, C_ROJO),
        ])

        # ── Tracking / ReID ──────────────────────────────────────────────────
        self._grupo(col_izq, "Tracking / ReID", f_group, [
            ("ReID umbral",  "reid_thr",    0.30, 0.95, 0.01, f_lbl, f_val, C_AZUL),
            ("Cooldown (s)", "cooldown_s",  1.0,  10.0, 0.5,  f_lbl, f_val, C_AZUL),
        ])

        # ── Beep ─────────────────────────────────────────────────────────────
        self._grupo(col_izq, "Beep de Cruce", f_group, [
            ("Frecuencia Hz","beep_freq",   200, 2000,  10,   f_lbl, f_val, C_AMARILLO),
            ("Duracion ms",  "beep_ms",     50,   500,  10,   f_lbl, f_val, C_AMARILLO),
        ])

        # ── Cámara ───────────────────────────────────────────────────────────
        self._grupo(col_der, "Camara — Imagen", f_group, [
            ("Brillo",       "cam_brillo",    -64,  64,   1,  f_lbl, f_val, C_AMARILLO),
            ("Contraste",    "cam_contraste",   0,  95,   1,  f_lbl, f_val, C_AMARILLO),
            ("Saturacion",   "cam_saturacion",  0, 100,   1,  f_lbl, f_val, C_AMARILLO),
            ("Exposicion",   "cam_exposicion", -13,  -1,  1,  f_lbl, f_val, C_AMARILLO),
        ])

        # ── Botones ───────────────────────────────────────────────────────────
        tk.Frame(r, bg=C_BORDE, height=1).pack(fill=tk.X, padx=10)
        btn_row = tk.Frame(r, bg=C_BG)
        btn_row.pack(pady=10)

        tk.Button(
            btn_row, text="Guardar",
            command=self._guardar,
            bg=C_VERDE, fg="#000",
            font=tkfont.Font(family="Arial", size=11, weight="bold"),
            relief=tk.FLAT, padx=20, pady=6, cursor="hand2",
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            btn_row, text="Restablecer defaults",
            command=self._restablecer,
            bg=C_GRIS, fg="#fff",
            font=tkfont.Font(family="Arial", size=11),
            relief=tk.FLAT, padx=20, pady=6, cursor="hand2",
        ).pack(side=tk.LEFT, padx=8)

        self._lbl_status = tk.Label(r, text="",
                                    bg=C_BG, fg=C_GRIS,
                                    font=tkfont.Font(family="Arial", size=9))
        self._lbl_status.pack(pady=(0, 6))

    def _grupo(
        self,
        parent,
        titulo: str,
        f_group,
        sliders: list,
    ) -> None:
        """Renderiza un grupo de sliders con título y borde."""
        frame = tk.LabelFrame(
            parent, text=f"  {titulo}  ",
            bg=C_GRUPO, fg=C_TEXTO,
            font=f_group,
            relief=tk.GROOVE, bd=1,
            labelanchor="n",
        )
        frame.pack(fill=tk.X, pady=6, ipadx=6, ipady=4)

        for (label, key, vmin, vmax, step, f_lbl, f_val, color) in sliders:
            self._slider(frame, label, key, vmin, vmax, step,
                         f_lbl, f_val, color)

    def _slider(
        self,
        parent,
        label: str,
        key:   str,
        vmin:  float,
        vmax:  float,
        step:  float,
        f_lbl,
        f_val,
        color: str,
    ) -> None:
        """Crea una fila: etiqueta | slider | valor actual."""
        row = tk.Frame(parent, bg=C_GRUPO)
        row.pack(fill=tk.X, padx=8, pady=3)

        tk.Label(row, text=f"{label}:", bg=C_GRUPO, fg=C_TEXTO,
                 font=f_lbl, width=14, anchor="w").pack(side=tk.LEFT)

        # Variable tk ligada al valor inicial de config
        is_int = isinstance(step, int) and isinstance(vmin, int)
        if is_int:
            var = tk.IntVar(value=int(self._cfg.get(key, DEFAULTS[key])))
        else:
            var = tk.DoubleVar(value=float(self._cfg.get(key, DEFAULTS[key])))
        self._vars[key] = var

        lbl_val = tk.Label(row, textvariable=var,
                           bg=C_GRUPO, fg=color,
                           font=f_val, width=6, anchor="e")
        lbl_val.pack(side=tk.RIGHT)

        scale = tk.Scale(
            row,
            variable=var,
            from_=vmin, to=vmax,
            resolution=step,
            orient=tk.HORIZONTAL,
            bg=C_GRUPO, fg=color,
            troughcolor="#2a2a2a",
            highlightthickness=0,
            showvalue=False,
            length=200,
            command=lambda val, k=key: self._on_slider(k, val),
        )
        scale.pack(side=tk.LEFT, padx=6)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_slider(self, key: str, raw_val: str) -> None:
        """Llamado por Tkinter cuando se mueve un slider."""
        try:
            val = float(raw_val)
            self._cfg[key] = val
            if self._on_change:
                self._on_change(key, val)
        except Exception as e:
            logger.warning("Error en slider %s: %s", key, e)

    def _guardar(self) -> None:
        self._save()
        self._lbl_status.config(text="Guardado correctamente.", fg=C_VERDE)
        self._root.after(2500, lambda: self._lbl_status.config(text=""))

    def _restablecer(self) -> None:
        self._cfg = dict(DEFAULTS)
        for key, var in self._vars.items():
            try:
                var.set(DEFAULTS[key])
                if self._on_change:
                    self._on_change(key, DEFAULTS[key])
            except Exception:
                pass
        self._lbl_status.config(text="Valores restablecidos.", fg=C_AMARILLO)
        self._root.after(2500, lambda: self._lbl_status.config(text=""))
