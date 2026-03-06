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

    def __init__(
        self,
        db,
        on_dorsal_confirmado: Optional[Callable] = None,
        on_inicio_carrera:    Optional[Callable] = None,
        on_fin_carrera:       Optional[Callable] = None,
    ) -> None:
        self._db                = db
        self._callback          = on_dorsal_confirmado
        self._on_inicio_carrera = on_inicio_carrera
        self._on_fin_carrera    = on_fin_carrera
        self._queue:   queue.Queue  = queue.Queue()
        self._pending: list         = []           # cola de cruces pendientes
        self._cruce:   Optional[dict] = None
        self._foto_img: Optional[ImageTk.PhotoImage] = None
        self._ocr_result: Optional[str] = None   # OCR del cruce actual

        threading.Thread(target=self._run, name="PopupCruce", daemon=True).start()

    def nuevo_cruce(
        self,
        id_registro:       int,
        track_id:          int,
        ts_ns:             int,
        foto_path:         Optional[str],
        con_casco:         bool,
        tiempo_carrera_ms: Optional[int] = None,
        dorsal_ocr:        Optional[str] = None,
    ) -> None:
        dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).astimezone()
        tiempo_str = dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"
        self._queue.put({
            "id_registro":       id_registro,
            "track_id":          track_id,
            "ts_ns":             ts_ns,
            "tiempo_str":        tiempo_str,
            "tiempo_carrera_ms": tiempo_carrera_ms,
            "foto_path":         foto_path,
            "con_casco":         con_casco,
            "dorsal_ocr":        dorsal_ocr,
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
        self._lbl_pendientes = tk.Label(hdr, text="", bg=C_PANEL,
                                        fg=C_AMARILLO, font=f_norm)
        self._lbl_pendientes.pack(side=tk.LEFT, padx=14)
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

        # ── Botón INICIO DE MARCAJE ───────────────────────────────────────────
        self._btn_inicio = tk.Button(
            col_der,
            text="INICIAR MARCAJE",
            command=self._iniciar_carrera,
            bg="#e65100", fg="white",
            font=tkfont.Font(family="Arial", size=20, weight="bold"),
            relief=tk.FLAT, padx=10, pady=18, cursor="hand2",
            activebackground="#bf360c", activeforeground="white",
        )
        self._btn_inicio.pack(fill=tk.X, padx=10, pady=(12, 4))

        self._lbl_estado_carrera = tk.Label(
            col_der, text="Carrera no iniciada",
            bg=C_PANEL, fg=C_GRIS,
            font=tkfont.Font(family="Arial", size=11),
        )
        self._lbl_estado_carrera.pack(pady=(0, 4))

        self._btn_fin = tk.Button(
            col_der,
            text="FINALIZAR CARRERA",
            command=self._finalizar_carrera,
            bg="#b71c1c", fg="white",
            font=tkfont.Font(family="Arial", size=14, weight="bold"),
            relief=tk.FLAT, padx=10, pady=10, cursor="hand2",
            activebackground="#7f0000", activeforeground="white",
            state=tk.DISABLED,
        )
        self._btn_fin.pack(fill=tk.X, padx=10, pady=(0, 6))

        tk.Frame(col_der, bg=C_BORDE, height=2).pack(fill=tk.X)

        tk.Label(col_der, text="PODIO EN VIVO", bg=C_PANEL,
                 fg=C_AMARILLO,
                 font=tkfont.Font(family="Arial", size=15, weight="bold")
                 ).pack(pady=10)
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

    def actualizar_ocr(self, id_registro: int, dorsal_ocr: Optional[str]) -> None:
        """Llamado desde el hilo OCR cuando Tesseract termina. Thread-safe."""
        self._queue.put({"_ocr_update": True, "id_registro": id_registro,
                         "dorsal_ocr": dorsal_ocr})

    def _poll(self) -> None:
        try:
            while True:
                msg = self._queue.get_nowait()
                if msg.get("_ocr_update"):
                    self._aplicar_ocr_update(msg)
                else:
                    self._pending.append(msg)
        except queue.Empty:
            pass
        # Si no hay cruce activo y hay pendientes, cargar el primero
        if self._cruce is None and self._pending:
            self._siguiente()
        self._actualizar_lbl_pendientes()
        self._root.after(100, self._poll)

    def _siguiente(self) -> None:
        """Carga el próximo cruce pendiente. Si no hay, limpia la pantalla."""
        if self._pending:
            self._cargar(self._pending.pop(0))
        else:
            self._cruce = None
            self._lbl_foto.config(image="", text="Esperando cruce...", fg=C_GRIS)
            self._foto_img = None
            self._var_dorsal.set("")
            self._lbl_nombre.config(text="← ingresá el dorsal", fg=C_GRIS)
            self._lbl_cat.config(text="")
            self._lbl_track.config(text="")
            self._lbl_tiempo.config(text="--:--:--.---")
            self._lbl_casco.config(text="", bg=C_PANEL)
            self._entry.config(bg="#222")
        self._actualizar_lbl_pendientes()

    def _actualizar_lbl_pendientes(self) -> None:
        n = len(self._pending)
        if n > 0:
            self._lbl_pendientes.config(text=f"  [{n} en cola]")
        else:
            self._lbl_pendientes.config(text="")

    def _aplicar_ocr_update(self, msg: dict) -> None:
        """Aplica el resultado OCR al cruce actual si el id_registro coincide."""
        if not self._cruce or self._cruce.get("id_registro") != msg["id_registro"]:
            return
        dorsal_ocr = msg.get("dorsal_ocr")
        if not dorsal_ocr:
            return
        self._ocr_result = dorsal_ocr
        # Solo pre-rellenar si el juez todavía no ingresó nada
        if not self._var_dorsal.get().strip():
            self._var_dorsal.set(dorsal_ocr)
            self._root.after(5, lambda: self._entry.config(bg="#3a2e00"))

    def _iniciar_carrera(self) -> None:
        if self._on_inicio_carrera:
            self._on_inicio_carrera()
        ts_ns = self._db.get_inicio_carrera()
        if ts_ns:
            dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).astimezone()
            hora_str = dt.strftime("%H:%M:%S")
            self._lbl_estado_carrera.config(
                text=f"En curso desde {hora_str}", fg=C_VERDE
            )
        self._btn_inicio.config(
            text="CARRERA EN CURSO",
            bg=C_VERDE, fg="#000",
            state=tk.DISABLED,
            cursor="arrow",
        )
        self._btn_fin.config(state=tk.NORMAL)

    def _finalizar_carrera(self) -> None:
        from tkinter import messagebox
        if not messagebox.askyesno(
            "Finalizar carrera",
            "¿Confirmar el cierre de la carrera?\n\n"
            "Se detendrá el marcaje y se exportarán los resultados a CSV.",
            icon="warning",
        ):
            return
        if self._on_fin_carrera:
            csv_path = self._on_fin_carrera()
        else:
            csv_path = None
        self._btn_fin.config(
            text="CARRERA FINALIZADA",
            bg=C_GRIS, fg="white",
            state=tk.DISABLED,
            cursor="arrow",
        )
        self._btn_inicio.config(state=tk.DISABLED)
        msg = "Carrera finalizada."
        if csv_path:
            msg += f"\nCSV: {csv_path}"
        self._lbl_estado_carrera.config(text="Carrera finalizada", fg=C_ROJO)
        messagebox.showinfo("Carrera finalizada", msg)

    @staticmethod
    def _fmt_carrera(ms: int) -> str:
        """Formatea milisegundos de carrera como MM:SS.mmm"""
        total_s  = ms // 1000
        resto_ms = ms % 1000
        minutos  = total_s // 60
        segundos = total_s % 60
        return f"{minutos:02d}:{segundos:02d}.{resto_ms:03d}"

    def _cargar(self, cruce: dict) -> None:
        self._cruce      = cruce
        self._ocr_result = cruce.get("dorsal_ocr")

        tc_ms = cruce.get("tiempo_carrera_ms")
        if tc_ms is not None:
            self._lbl_tiempo.config(text=self._fmt_carrera(tc_ms))
        else:
            self._lbl_tiempo.config(text=cruce["tiempo_str"])
        self._lbl_track.config(text=f"  Track #{cruce['track_id']}")

        if cruce["con_casco"]:
            self._lbl_casco.config(text="✅ Con casco", fg=C_VERDE, bg=C_PANEL)
        else:
            self._lbl_casco.config(text="❌ Sin casco", fg=C_ROJO,  bg=C_PANEL)

        self._mostrar_foto(cruce.get("foto_path"))

        # Pre-rellenar con resultado OCR si existe
        if self._ocr_result:
            self._var_dorsal.set(self._ocr_result)
            # _on_dorsal_change se dispara por el trace; forzamos fondo ámbar
            self._root.after(5, lambda: self._entry.config(bg="#3a2e00"))
        else:
            self._var_dorsal.set("")
            self._lbl_nombre.config(text="← ingresá el dorsal", fg=C_GRIS)
            self._lbl_cat.config(text="")
            self._entry.config(bg="#222")

        self._entry.focus_set()
        self._actualizar_podio()
        self._root.lift()   # trae al frente sin robar el foco de otras ventanas

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
        dorsal    = self._var_dorsal.get().strip()
        is_ocr    = bool(dorsal and dorsal == self._ocr_result)

        if not dorsal:
            self._lbl_nombre.config(text="← ingresá el dorsal", fg=C_GRIS)
            self._lbl_cat.config(text="")
            self._entry.config(bg="#222")
            return

        atleta = self._db.buscar_atleta(dorsal)
        if atleta:
            prefijo = "  [OCR] " if is_ocr else "  "
            self._lbl_nombre.config(
                text=f"{prefijo}{atleta['nombre']}", fg=C_VERDE)
            self._lbl_cat.config(
                text=f"  {atleta['categoria']}  {atleta['equipo']}", fg=C_AZUL)
            self._entry.config(bg="#1a3a1a" if not is_ocr else "#2a3a00")
        elif is_ocr:
            self._lbl_nombre.config(
                text=f"  OCR: #{dorsal}  (sin registro en BD)", fg=C_AMARILLO)
            self._lbl_cat.config(text="")
            self._entry.config(bg="#3a2e00")   # fondo ámbar
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
        self._root.after(1200, lambda: (
            self._btn_ok.config(text="✓  CONFIRMAR  (Enter)", bg=C_VERDE),
            self._siguiente(),
        ))

    def _cancelar(self) -> None:
        self._siguiente()

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
                if c["tiempo_carrera_ms"] is not None:
                    tiempo_display = self._fmt_carrera(c["tiempo_carrera_ms"])
                else:
                    raw = c["tiempo_cruce"]
                    tiempo_display = raw.split("T")[-1][:12] if "T" in raw else raw[:12]
                nombre = c["nombre"][:14]
                t.insert(tk.END,
                         f"  {icono} #{c['dorsal']:<4} {nombre:<14} {tiempo_display}\n", tag)

        t.config(state=tk.DISABLED)
