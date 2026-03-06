"""
admin.py — Gestión de atletas y resultados del evento.

Ventana de administración independiente:
  • Importar atletas desde Excel (.xlsx) o CSV
  • Ver, editar y eliminar atletas
  • Ver resultados / podio del evento

Uso:
    python3 admin.py
    python3 admin.py --db data/carreras.db
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk
from pathlib import Path
from typing import Optional

# Agregar el directorio del proyecto al path
sys.path.insert(0, str(Path(__file__).parent))
from utils.db_manager import DBManager

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# ─── Colores ──────────────────────────────────────────────────────────────────
C_BG      = "#0f0f0f"
C_PANEL   = "#1a1a1a"
C_VERDE   = "#00e676"
C_AMARILLO= "#ffd600"
C_ROJO    = "#ff1744"
C_AZUL    = "#29b6f6"
C_TEXTO   = "#f0f0f0"
C_GRIS    = "#777777"
C_BORDE   = "#333333"
C_SEL     = "#1e3a5f"


class AdminApp:

    def __init__(self, db: DBManager) -> None:
        self._db = db
        self._root = tk.Tk()
        self._root.title("⚙  Administración — Sistema de Cronometraje")
        self._root.geometry("1100x700")
        self._root.configure(bg=C_BG)
        self._root.resizable(True, True)

        self._build_ui()
        self._cargar_tabla_atletas()

    def run(self) -> None:
        self._root.mainloop()

    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        r = self._root

        f_titulo = tkfont.Font(family="Arial", size=13, weight="bold")
        f_norm   = tkfont.Font(family="Arial", size=11)
        f_small  = tkfont.Font(family="Arial", size=10)
        f_mono   = tkfont.Font(family="Courier", size=10)

        # ── Notebook (pestañas) ───────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook",       background=C_BG,    borderwidth=0)
        style.configure("TNotebook.Tab",   background=C_PANEL, foreground=C_GRIS,
                         padding=[14, 6], font=f_norm)
        style.map("TNotebook.Tab",
                  background=[("selected", C_AZUL)],
                  foreground=[("selected", "#000")])
        style.configure("Treeview",
                         background=C_PANEL, foreground=C_TEXTO,
                         fieldbackground=C_PANEL, rowheight=24,
                         font=f_norm)
        style.configure("Treeview.Heading",
                         background=C_BORDE, foreground=C_AMARILLO,
                         font=f_norm)
        style.map("Treeview", background=[("selected", C_SEL)])

        nb = ttk.Notebook(r)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        tab_atletas   = tk.Frame(nb, bg=C_BG)
        tab_resultados = tk.Frame(nb, bg=C_BG)

        nb.add(tab_atletas,    text="  👥  Atletas  ")
        nb.add(tab_resultados, text="  🏆  Resultados / Podio  ")

        nb.bind("<<NotebookTabChanged>>",
                lambda e: self._on_tab_change(nb.index(nb.select())))

        self._build_tab_atletas(tab_atletas, f_titulo, f_norm, f_small, f_mono)
        self._build_tab_resultados(tab_resultados, f_titulo, f_norm, f_mono)

    # ──────────────────────────────────────────────────────────────────────────
    #  Pestaña Atletas
    # ──────────────────────────────────────────────────────────────────────────

    def _build_tab_atletas(self, tab, f_titulo, f_norm, f_small, f_mono) -> None:

        # ── Barra superior: importar + buscar ─────────────────────────────────
        top = tk.Frame(tab, bg=C_PANEL, pady=8)
        top.pack(fill=tk.X, padx=6, pady=(6,0))

        # Botón importar Excel
        tk.Button(
            top, text="📂  Importar Excel / CSV",
            command=self._importar_archivo,
            bg=C_VERDE, fg="#000",
            font=tkfont.Font(family="Arial", size=12, weight="bold"),
            relief=tk.FLAT, padx=16, pady=6, cursor="hand2",
        ).pack(side=tk.LEFT, padx=10)

        # Formato esperado
        tk.Label(top,
                 text="Columnas Excel: Dorsal | Nombre | Categoria | Equipo  (encabezado en fila 1)",
                 bg=C_PANEL, fg=C_GRIS, font=f_small,
                 ).pack(side=tk.LEFT, padx=10)

        # Buscador
        tk.Label(top, text="Buscar:", bg=C_PANEL,
                 fg=C_TEXTO, font=f_norm).pack(side=tk.RIGHT, padx=(0,4))
        self._var_buscar = tk.StringVar()
        self._var_buscar.trace_add("write", lambda *_: self._filtrar_tabla())
        tk.Entry(top, textvariable=self._var_buscar,
                 bg="#222", fg=C_TEXTO, insertbackground=C_TEXTO,
                 relief=tk.FLAT, font=f_norm, width=18, bd=3,
                 ).pack(side=tk.RIGHT, padx=10)

        # ── Tabla de atletas ──────────────────────────────────────────────────
        tabla_frame = tk.Frame(tab, bg=C_BG)
        tabla_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        cols = ("dorsal", "nombre", "categoria", "equipo")
        self._tree = ttk.Treeview(tabla_frame, columns=cols,
                                   show="headings", selectmode="browse")

        self._tree.heading("dorsal",    text="Dorsal")
        self._tree.heading("nombre",    text="Nombre")
        self._tree.heading("categoria", text="Categoría")
        self._tree.heading("equipo",    text="Equipo")

        self._tree.column("dorsal",    width=80,  anchor="center")
        self._tree.column("nombre",    width=250)
        self._tree.column("categoria", width=140)
        self._tree.column("equipo",    width=180)

        sb = ttk.Scrollbar(tabla_frame, orient=tk.VERTICAL,
                           command=self._tree.yview)
        self._tree.configure(yscroll=sb.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._tree.bind("<Double-1>", lambda e: self._editar_seleccionado())

        # ── Panel inferior: agregar / editar / eliminar ───────────────────────
        bottom = tk.Frame(tab, bg=C_PANEL, pady=10)
        bottom.pack(fill=tk.X, padx=6, pady=(0,6))

        campos = [("Dorsal", 8), ("Nombre", 28), ("Categoría", 18), ("Equipo", 18)]
        self._vars_form = {}
        for i, (lbl, w) in enumerate(campos):
            tk.Label(bottom, text=lbl + ":", bg=C_PANEL,
                     fg=C_TEXTO, font=f_norm).grid(row=0, column=i*2, padx=(12,2), pady=6)
            var = tk.StringVar()
            self._vars_form[lbl.lower().replace("í","i")] = var
            tk.Entry(bottom, textvariable=var, width=w,
                     bg="#222", fg=C_VERDE, insertbackground=C_VERDE,
                     relief=tk.FLAT, font=f_norm, bd=3,
                     ).grid(row=0, column=i*2+1, padx=(0,6), pady=6)

        # Botones del form
        btn_frame = tk.Frame(bottom, bg=C_PANEL)
        btn_frame.grid(row=0, column=8, padx=10)

        tk.Button(btn_frame, text="➕ Agregar",
                  command=self._agregar_atleta,
                  bg=C_AZUL, fg="#000",
                  font=tkfont.Font(family="Arial", size=11, weight="bold"),
                  relief=tk.FLAT, padx=12, pady=5, cursor="hand2",
                  ).pack(side=tk.LEFT, padx=4)

        tk.Button(btn_frame, text="✏ Guardar",
                  command=self._guardar_edicion,
                  bg=C_AMARILLO, fg="#000",
                  font=tkfont.Font(family="Arial", size=11, weight="bold"),
                  relief=tk.FLAT, padx=12, pady=5, cursor="hand2",
                  ).pack(side=tk.LEFT, padx=4)

        tk.Button(btn_frame, text="🗑 Eliminar",
                  command=self._eliminar_seleccionado,
                  bg=C_ROJO, fg="white",
                  font=tkfont.Font(family="Arial", size=11, weight="bold"),
                  relief=tk.FLAT, padx=12, pady=5, cursor="hand2",
                  ).pack(side=tk.LEFT, padx=4)

        # Contador
        self._lbl_total = tk.Label(tab, text="", bg=C_BG,
                                   fg=C_GRIS, font=f_small)
        self._lbl_total.pack(anchor="w", padx=10, pady=(0,4))

        # Cuando se selecciona una fila, cargar en el form
        self._tree.bind("<<TreeviewSelect>>", self._on_seleccionar)

    # ──────────────────────────────────────────────────────────────────────────
    #  Pestaña Resultados
    # ──────────────────────────────────────────────────────────────────────────

    def _build_tab_resultados(self, tab, f_titulo, f_norm, f_mono) -> None:

        top = tk.Frame(tab, bg=C_PANEL, pady=8)
        top.pack(fill=tk.X, padx=6, pady=(6,0))

        tk.Label(top, text="🏆  Podio y resultados del evento",
                 bg=C_PANEL, fg=C_AMARILLO, font=f_titulo).pack(side=tk.LEFT, padx=12)

        tk.Button(top, text="↺  Actualizar",
                  command=self._cargar_resultados,
                  bg=C_BORDE, fg=C_TEXTO,
                  font=f_norm, relief=tk.FLAT, padx=12, pady=4, cursor="hand2",
                  ).pack(side=tk.RIGHT, padx=10)

        tk.Button(top, text="📋  Exportar CSV",
                  command=self._exportar_csv,
                  bg=C_VERDE, fg="#000",
                  font=tkfont.Font(family="Arial", size=11, weight="bold"),
                  relief=tk.FLAT, padx=12, pady=4, cursor="hand2",
                  ).pack(side=tk.RIGHT, padx=4)

        # Tabla resultados
        cols = ("puesto", "dorsal", "nombre", "categoria",
                "equipo", "tiempo", "casco")
        self._tree_res = ttk.Treeview(tab, columns=cols,
                                       show="headings", selectmode="browse")

        anchos = [70, 80, 240, 140, 160, 130, 90]
        hdrs   = ["Puesto", "Dorsal", "Nombre", "Categoría",
                  "Equipo", "Tiempo", "Casco"]
        for col, hdr, ancho in zip(cols, hdrs, anchos):
            self._tree_res.heading(col, text=hdr)
            self._tree_res.column(col, width=ancho,
                                   anchor="center" if col in ("puesto","dorsal","casco") else "w")

        self._tree_res.tag_configure("p1",  background="#2a2200")
        self._tree_res.tag_configure("p2",  background="#1a1a1a")
        self._tree_res.tag_configure("p3",  background="#1a1000")
        self._tree_res.tag_configure("cat", background="#0d1f0d")

        sb2 = ttk.Scrollbar(tab, orient=tk.VERTICAL,
                             command=self._tree_res.yview)
        self._tree_res.configure(yscroll=sb2.set)

        self._tree_res.pack(side=tk.LEFT, fill=tk.BOTH,
                             expand=True, padx=(6,0), pady=6)
        sb2.pack(side=tk.RIGHT, fill=tk.Y, pady=6, padx=(0,6))

    # ──────────────────────────────────────────────────────────────────────────
    #  Importar Excel / CSV
    # ──────────────────────────────────────────────────────────────────────────

    def _importar_archivo(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleccioná el archivo de atletas",
            filetypes=[
                ("Excel y CSV", "*.xlsx *.xls *.csv"),
                ("Excel",       "*.xlsx *.xls"),
                ("CSV",         "*.csv"),
                ("Todos",       "*.*"),
            ],
        )
        if not path:
            return

        try:
            filas = self._leer_archivo(path)
        except Exception as e:
            messagebox.showerror("Error al leer archivo",
                                 f"No se pudo leer el archivo:\n{e}")
            return

        if not filas:
            messagebox.showwarning("Archivo vacío",
                                   "No se encontraron datos en el archivo.")
            return

        # Vista previa y confirmación
        preview = "\n".join(
            f"  {f['dorsal']:<6}  {f['nombre']:<25}  {f['categoria']}"
            for f in filas[:8]
        )
        if len(filas) > 8:
            preview += f"\n  ... y {len(filas)-8} más"

        ok = messagebox.askyesno(
            "Confirmar importación",
            f"Se van a importar {len(filas)} atletas:\n\n{preview}\n\n"
            f"Los dorsales existentes serán actualizados.\n¿Continuar?",
        )
        if not ok:
            return

        errores = 0
        for f in filas:
            try:
                self._db.registrar_atleta(
                    dorsal    = str(f["dorsal"]).strip(),
                    nombre    = str(f["nombre"]).strip(),
                    categoria = str(f["categoria"]).strip(),
                    equipo    = str(f.get("equipo", "")).strip(),
                )
            except Exception:
                errores += 1

        self._cargar_tabla_atletas()
        msg = f"✅ {len(filas)-errores} atletas importados correctamente."
        if errores:
            msg += f"\n⚠ {errores} filas con error fueron omitidas."
        messagebox.showinfo("Importación completada", msg)

    def _leer_archivo(self, path: str) -> list[dict]:
        """Lee Excel (.xlsx) o CSV y devuelve lista de dicts."""
        p = Path(path)

        if p.suffix.lower() in (".xlsx", ".xls"):
            return self._leer_excel(p)
        else:
            return self._leer_csv(p)

    def _leer_excel(self, path: Path) -> list[dict]:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        wb.close()
        if not rows:
            return []

        # Primera fila = encabezados
        headers = [str(c).strip().lower() if c else "" for c in rows[0]]
        col_map = self._mapear_columnas(headers)

        result = []
        for row in rows[1:]:
            if all(c is None or str(c).strip() == "" for c in row):
                continue
            d = self._fila_a_dict(row, col_map)
            if d:
                result.append(d)
        return result

    def _leer_csv(self, path: Path) -> list[dict]:
        # Detectar separador (coma o punto y coma)
        raw = path.read_text(encoding="utf-8-sig", errors="replace")
        sep = ";" if raw.count(";") > raw.count(",") else ","

        reader = csv.DictReader(io.StringIO(raw), delimiter=sep)
        headers = [k.strip().lower() for k in (reader.fieldnames or [])]
        col_map = self._mapear_columnas(headers)

        result = []
        for row in reader:
            values = list(row.values())
            d = self._fila_a_dict(values, col_map)
            if d:
                result.append(d)
        return result

    @staticmethod
    def _mapear_columnas(headers: list[str]) -> dict[str, int]:
        """Mapea nombres de columna a índice, tolerando variaciones."""
        mapa = {}
        sinonimos = {
            "dorsal":    ["dorsal", "nro", "número", "numero", "num", "bib", "#"],
            "nombre":    ["nombre", "name", "atleta", "corredor", "apellido y nombre",
                          "apellido_nombre", "nombre y apellido"],
            "categoria": ["categoria", "categoría", "cat", "category", "grupo",
                          "division", "división"],
            "equipo":    ["equipo", "team", "club", "escuadra"],
        }
        for campo, variantes in sinonimos.items():
            for i, h in enumerate(headers):
                if any(v in h for v in variantes):
                    mapa[campo] = i
                    break
        return mapa

    @staticmethod
    def _fila_a_dict(row, col_map: dict) -> Optional[dict]:
        """Convierte una fila a dict con las claves estándar."""
        def get(key):
            idx = col_map.get(key)
            if idx is None or idx >= len(row):
                return ""
            v = row[idx]
            return str(v).strip() if v is not None else ""

        dorsal = get("dorsal")
        nombre = get("nombre")
        if not dorsal or not nombre:
            return None
        return {
            "dorsal":    dorsal,
            "nombre":    nombre,
            "categoria": get("categoria") or "General",
            "equipo":    get("equipo"),
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  Tabla de atletas
    # ──────────────────────────────────────────────────────────────────────────

    def _cargar_tabla_atletas(self) -> None:
        self._tree.delete(*self._tree.get_children())
        atletas = self._db.get_todos_atletas()
        for a in atletas:
            self._tree.insert("", tk.END, iid=a["dorsal"],
                               values=(a["dorsal"], a["nombre"],
                                       a["categoria"], a["equipo"]))
        n = len(atletas)
        self._lbl_total.config(text=f"  {n} atleta{'s' if n!=1 else ''} registrado{'s' if n!=1 else ''}")

    def _filtrar_tabla(self) -> None:
        texto = self._var_buscar.get().strip().lower()
        self._tree.delete(*self._tree.get_children())
        atletas = self._db.get_todos_atletas()
        for a in atletas:
            if texto in (a["dorsal"]+a["nombre"]+a["categoria"]+a["equipo"]).lower():
                self._tree.insert("", tk.END, iid=a["dorsal"],
                                   values=(a["dorsal"], a["nombre"],
                                           a["categoria"], a["equipo"]))

    def _on_seleccionar(self, _event=None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        vals = self._tree.item(sel[0], "values")
        self._vars_form["dorsal"].set(vals[0])
        self._vars_form["nombre"].set(vals[1])
        self._vars_form["categoria"].set(vals[2])
        self._vars_form["equipo"].set(vals[3])

    def _editar_seleccionado(self) -> None:
        self._on_seleccionar()

    def _agregar_atleta(self) -> None:
        d = self._vars_form["dorsal"].get().strip()
        n = self._vars_form["nombre"].get().strip()
        c = self._vars_form["categoria"].get().strip()
        e = self._vars_form["equipo"].get().strip()
        if not d or not n:
            messagebox.showwarning("Datos incompletos",
                                   "Dorsal y Nombre son obligatorios.")
            return
        self._db.registrar_atleta(d, n, c or "General", e)
        self._cargar_tabla_atletas()
        for v in self._vars_form.values():
            v.set("")

    def _guardar_edicion(self) -> None:
        sel = self._tree.selection()
        if not sel:
            messagebox.showinfo("Selección", "Seleccioná un atleta para editar.")
            return
        self._agregar_atleta()   # INSERT OR REPLACE hace la actualización

    def _eliminar_seleccionado(self) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        dorsal = sel[0]
        nombre = self._tree.item(dorsal, "values")[1]
        if not messagebox.askyesno("Confirmar",
                                   f"¿Eliminar a {nombre} (dorsal {dorsal})?"):
            return
        self._db._conn.execute(
            "DELETE FROM atletas WHERE dorsal = ?", (dorsal,))
        self._db._conn.commit()
        self._cargar_tabla_atletas()

    # ──────────────────────────────────────────────────────────────────────────
    #  Resultados
    # ──────────────────────────────────────────────────────────────────────────

    def _on_tab_change(self, idx: int) -> None:
        if idx == 1:
            self._cargar_resultados()

    def _cargar_resultados(self) -> None:
        self._tree_res.delete(*self._tree_res.get_children())
        rows = self._db.get_podio()

        cat_actual = None
        pos_en_cat = 0
        medallas = ["🥇", "🥈", "🥉"]

        for r in rows:
            cat = r["categoria"]
            if cat != cat_actual:
                cat_actual = cat
                pos_en_cat = 0
                self._tree_res.insert("", tk.END,
                                       values=("", "", f"── {cat} ──",
                                               "", "", "", ""),
                                       tags=("cat",))
            pos_en_cat += 1
            icono = medallas[pos_en_cat-1] if pos_en_cat <= 3 else f"{pos_en_cat}°"
            hora  = r["tiempo_cruce"].split("T")[-1] if "T" in r["tiempo_cruce"] \
                    else r["tiempo_cruce"]
            tag   = ("p1" if pos_en_cat==1 else
                     "p2" if pos_en_cat==2 else
                     "p3" if pos_en_cat==3 else "")
            self._tree_res.insert("", tk.END,
                                   values=(icono, r["dorsal"], r["nombre"],
                                           r["categoria"], r["equipo"],
                                           hora,
                                           "Sí" if r["con_casco"] else "No"),
                                   tags=(tag,) if tag else ())

    def _exportar_csv(self) -> None:
        rows = self._db.get_podio()
        if not rows:
            messagebox.showinfo("Sin datos", "No hay resultados para exportar.")
            return

        path = filedialog.asksaveasfilename(
            title="Guardar resultados",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Todos", "*.*")],
            initialfile="resultados_evento.csv",
        )
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["Puesto", "Dorsal", "Nombre", "Categoria",
                         "Equipo", "Tiempo", "Con_casco"])
            cat_actual = None
            pos = 0
            for r in rows:
                if r["categoria"] != cat_actual:
                    cat_actual = r["categoria"]
                    pos = 0
                pos += 1
                hora = r["tiempo_cruce"].split("T")[-1] if "T" in r["tiempo_cruce"] \
                       else r["tiempo_cruce"]
                w.writerow([pos, r["dorsal"], r["nombre"],
                             r["categoria"], r["equipo"], hora,
                             "Sí" if r["con_casco"] else "No"])

        messagebox.showinfo("Exportado", f"Resultados guardados en:\n{path}")


# ─── Entrada ──────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Administración del sistema de cronometraje")
    p.add_argument("--db", default="data/carreras.db",
                   help="Ruta a la base de datos SQLite")
    args = p.parse_args()

    db = DBManager(args.db)
    app = AdminApp(db)
    app.run()
    db.close()


if __name__ == "__main__":
    main()
