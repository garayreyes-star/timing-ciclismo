"""
popup_cruce.py — Ventana Tkinter para registro de dorsales en tiempo real.
"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from tkinter import font as tkfont
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

from PIL import Image, ImageTk

logger = logging.getLogger(__name__)

C_BG       = "#0d0d0d"
C_PANEL    = "#1a1a1a"
C_VERDE    = "#00e676"
C_AMARILLO = "#ffd600"
C_ROJO     = "#ff1744"
C_TEXTO    = "#f0f0f0"
C_GRIS     = "#777777"
C_AZUL     = "#29b6f6"
C_BORDE    = "#333333"


class PopupCruce:

    def __init__(self, db, on_dorsal_confirmado: Optional[Callable] = None) -> None:
        self._db       = db
        self._callback = on_dorsal_confirmado
        self._queue:   queue.Queue = queue.Queue()
        self._cruce:   Optional[dict] = None
        self._foto_img: Optional[ImageTk.PhotoImage] = None

        threading.Thread(target=self._run, name="PopupCruce", daemon=True).start()

    def nuevo_cruce(self, id_registro: int, track_id: int, ts_ns: int,
                    foto_path: Optional[str], con_casco: bool) -> None:
        dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).astimezone()
        tiempo_str = dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"
        self._queue.put({
            "id_registro": id_registro,
            "track_id":    track_id,
            "ts_ns":       ts_ns,
            "tiempo_str":  tiempo_str,
            "foto_path":   foto_path,
            "con_casco":   con_casco,
        })

    # ── Hilo Tkinter ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        self._root = tk.Tk()
        self._root.title("Meta — Registro de Dorsales")
        self._root.configure(bg=C_BG)
        self._root.geometry("1400x900")

        self._build_ui()
        self._poll()
        self._root.mainloop()

    def _build_ui(self) -> None:
        r = self._root

        f_big   = tkfont.Font(family="Arial",  size=32, weight="bold")
        f_med   = tkfont.Font(family="Arial",  size=16, weight="bold")
        f_norm  = tkfont.Font(family="Arial",  size=13)
        f_mono  = tkfont.Font(family="Courier",size=11)
        f_small = tkfont.Font(family="Arial",  size=10)
        self._f_podio = f_mono

        # ── Columna izquierda (fija 980px) y derecha (podio) ─────────────────
        col_izq = tk.Frame(r, bg=C_BG, width=980)
        col_der = tk.Frame(r, bg=C_PANEL, width=400)
        col_izq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        col_der.pack(side=tk.RIGHT, fill=tk.Y)
        col_der.pack_propagate(False)

        # ─────────────────── COLUMNA IZQUIERDA ───────────────────────────────

        # 1. Encabezado: tiempo + track + casco
        hdr = tk.Frame(col_izq, bg=C_PANEL, pady=6)
        hdr.pack(fill=tk.X, padx=6, pady=(6,0))

        tk.Label(hdr, text="🚴  CRUCE", bg=C_PANEL,
                 fg=C_VERDE, font=f_med).pack(side=tk.LEFT, padx=12)
        self._lbl_track = tk.Label(hdr, text="", bg=C_PANEL,
                                   fg=C_GRIS, font=f_norm)
        self._lbl_track.pack(side=tk.LEFT)
        self._lbl_casco = tk.Label(hdr, text="", bg=C_PANEL, font=f_norm)
        self._lbl_casco.pack(side=tk.RIGHT, padx=12)
        self._lbl_tiempo = tk.Label(hdr, text="--:--:--.---", bg=C_PANEL,
                                    fg=C_AMARILLO, font=f_big)
        self._lbl_tiempo.pack(side=tk.RIGHT, padx=20)

        # 2. Foto (altura fija 580px)
        foto_cont = tk.Frame(col_izq, bg="#111", height=580)
        foto_cont.pack(fill=tk.X, padx=6, pady=6)
        foto_cont.pack_propagate(False)

        self._lbl_foto = tk.Label(foto_cont, bg="#111",
                                  text="Esperando cruce...",
                                  fg=C_GRIS, font=f_med)
        self._lbl_foto.pack(fill=tk.BOTH, expand=True)

        # 3. Sección dorsal
        sec = tk.Frame(col_izq, bg=C_PANEL, pady=10)
        sec.pack(fill=tk.X, padx=6, pady=(0,4))

        tk.Label(sec, text="Dorsal:", bg=C_PANEL,
                 fg=C_TEXTO, font=f_med).grid(row=0, column=0, padx=14, sticky="w")

        self._var_dorsal = tk.StringVar()
        self._var_dorsal.trace_add("write", self._on_dorsal_change)
        self._entry = tk.Entry(
            sec, textvariable=self._var_dorsal,
            font=tkfont.Font(family="Arial", size=26, weight="bold"),
            width=7, bg="#222", fg=C_VERDE,
            insertbackground=C_VERDE, relief=tk.FLAT,
            justify=tk.CENTER, bd=4,
        )
        self._entry.grid(row=0, column=1, padx=8, pady=6)
        self._entry.bind("<Return>", lambda e: self._confirmar())

        self._lbl_nombre = tk.Label(sec, text="← ingresá el dorsal",
                                    bg=C_PANEL, fg=C_GRIS,
                                    font=f_med, anchor="w")
        self._lbl_nombre.grid(row=0, column=2, padx=10, sticky="w")

        self._lbl_cat = tk.Label(sec, text="",
                                 bg=C_PANEL, fg=C_AZUL,
                                 font=f_norm, anchor="w")
        self._lbl_cat.grid(row=1, column=2, padx=10, sticky="w")

        # 4. Botones
        btns = tk.Frame(col_izq, bg=C_BG)
        btns.pack(pady=8)

        self._btn_ok = tk.Button(
            btns, text="✓  CONFIRMAR  (Enter)",
            command=self._confirmar,
            bg=C_VERDE, fg="#000",
            font=tkfont.Font(family="Arial", size=14, weight="bold"),
            relief=tk.FLAT, padx=24, pady=10, cursor="hand2",
        )
        self._btn_ok.pack(side=tk.LEFT, padx=12)

        tk.Button(
            btns, text="✗  CANCELAR  (Esc)",
            command=self._cancelar,
            bg=C_ROJO, fg="white",
            font=tkfont.Font(family="Arial", size=14, weight="bold"),
            relief=tk.FLAT, padx=24, pady=10, cursor="hand2",
        ).pack(side=tk.LEFT, padx=12)

        r.bind("<Escape>", lambda e: self._cancelar())

        # ─────────────────── COLUMNA DERECHA — PODIO ─────────────────────────

        tk.Label(col_der, text="🏆  PODIO EN VIVO", bg=C_PANEL,
                 fg=C_AMARILLO,
                 font=tkfont.Font(family="Arial", size=15, weight="bold")
                 ).pack(pady=12)
        tk.Frame(col_der, bg=C_BORDE, height=1).pack(fill=tk.X)

        self._podio_txt = tk.Text(
            col_der, bg=C_PANEL, fg=C_TEXTO,
            font=f_mono, relief=tk.FLAT,
            state=tk.DISABLED, wrap=tk.NONE,
            padx=10, pady=6,
        )
        self._podio_txt.pack(fill=tk.BOTH, expand=True)
        self._podio_txt.tag_config("cat",   foreground=C_AMARILLO,
                                   font=tkfont.Font(family="Arial", size=11, weight="bold"))
        self._podio_txt.tag_config("p1",    foreground="#ffd700")
        self._podio_txt.tag_config("p2",    foreground="#c0c0c0")
        self._podio_txt.tag_config("p3",    foreground="#cd7f32")
        self._podio_txt.tag_config("resto", foreground=C_TEXTO)
        self._podio_txt.tag_config("sep",   foreground=C_BORDE)

        tk.Label(r, text="ENTER = confirmar  |  ESC = cancelar",
                 bg=C_BG, fg=C_GRIS,
                 font=f_small).pack(side=tk.BOTTOM, pady=3)

    # ── Polling ───────────────────────────────────────────────────────────────

    def _poll(self) -> None:
        try:
            while True:
                self._cargar(self._queue.get_nowait())
        except queue.Empty:
            pass
        self._root.after(100, self._poll)

    def _cargar(self, cruce: dict) -> None:
        self._cruce = cruce
        self._lbl_tiempo.config(text=cruce["tiempo_str"])
        self._lbl_track.config(text=f"  Track #{cruce['track_id']}")

        if cruce["con_casco"]:
            self._lbl_casco.config(text="✅ Con casco", fg=C_VERDE, bg=C_PANEL)
        else:
            self._lbl_casco.config(text="❌ Sin casco", fg=C_ROJO,  bg=C_PANEL)

        self._mostrar_foto(cruce.get("foto_path"))
        self._var_dorsal.set("")
        self._lbl_nombre.config(text="← ingresá el dorsal", fg=C_GRIS)
        self._lbl_cat.config(text="")
        self._entry.config(bg="#222")
        self._entry.focus_set()
        self._actualizar_podio()
        self._root.lift()
        self._root.focus_force()

    def _mostrar_foto(self, foto_path: Optional[str]) -> None:
        if not foto_path or not Path(foto_path).exists():
            self._lbl_foto.config(image="", text="Sin foto", fg=C_GRIS)
            self._foto_img = None
            return
        try:
            img = Image.open(foto_path)
            img.thumbnail((960, 575), Image.LANCZOS)
            self._foto_img = ImageTk.PhotoImage(img)
            self._lbl_foto.config(image=self._foto_img, text="")
        except Exception as e:
            logger.warning("Error foto: %s", e)
            self._lbl_foto.config(image="", text="Error al cargar foto", fg=C_ROJO)

    # ── Dorsal ────────────────────────────────────────────────────────────────

    def _on_dorsal_change(self, *_) -> None:
        dorsal = self._var_dorsal.get().strip()
        if not dorsal:
            self._lbl_nombre.config(text="← ingresá el dorsal", fg=C_GRIS)
            self._lbl_cat.config(text="")
            self._entry.config(bg="#222")
            return
        atleta = self._db.buscar_atleta(dorsal)
        if atleta:
            self._lbl_nombre.config(text=f"  {atleta['nombre']}", fg=C_VERDE)
            self._lbl_cat.config(
                text=f"  {atleta['categoria']}  {atleta['equipo']}", fg=C_AZUL)
            self._entry.config(bg="#1a3a1a")
        else:
            self._lbl_nombre.config(text="  No registrado", fg=C_ROJO)
            self._lbl_cat.config(text="")
            self._entry.config(bg="#3a1a1a")

    def _confirmar(self) -> None:
        if not self._cruce:
            return
        dorsal = self._var_dorsal.get().strip()
        if not dorsal:
            self._entry.config(bg="#5a1a1a")
            return
        id_reg = self._cruce["id_registro"]
        self._db.actualizar_dorsal(id_reg, dorsal)
        logger.info("Dorsal confirmado: id=%d dorsal=%s", id_reg, dorsal)
        if self._callback:
            self._callback(id_reg, dorsal)
        self._actualizar_podio()
        self._btn_ok.config(text="✓  GUARDADO ✓", bg="#00c853")
        self._root.after(1800, lambda: self._btn_ok.config(
            text="✓  CONFIRMAR  (Enter)", bg=C_VERDE))

    def _cancelar(self) -> None:
        self._var_dorsal.set("")
        self._lbl_nombre.config(text="  Cancelado", fg=C_GRIS)
        self._lbl_cat.config(text="")

    # ── Podio ─────────────────────────────────────────────────────────────────

    def _actualizar_podio(self) -> None:
        rows = self._db.get_podio()
        t = self._podio_txt
        t.config(state=tk.NORMAL)
        t.delete("1.0", tk.END)

        if not rows:
            t.insert(tk.END, "\n  Aún sin cruces registrados.\n", "resto")
            t.config(state=tk.DISABLED)
            return

        cats: dict[str, list] = {}
        for r in rows:
            cats.setdefault(r["categoria"], []).append(r)

        medals = ["🥇", "🥈", "🥉"]
        tags   = ["p1", "p2", "p3"]

        for cat, lista in cats.items():
            t.insert(tk.END, f"\n  {cat}\n", "cat")
            t.insert(tk.END, "  " + "─"*36 + "\n", "sep")
            for i, c in enumerate(lista):
                icono = medals[i] if i < 3 else f"  {i+1:2d}."
                tag   = tags[i]   if i < 3 else "resto"
                hora  = c["tiempo_cruce"].split("T")[-1] if "T" in c["tiempo_cruce"] \
                        else c["tiempo_cruce"]
                nombre = c["nombre"][:14]
                t.insert(tk.END,
                         f"  {icono} #{c['dorsal']:<4} {nombre:<14} {hora}\n", tag)

        t.config(state=tk.DISABLED)
