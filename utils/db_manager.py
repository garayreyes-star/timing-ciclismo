"""
db_manager.py — Gestor de base de datos para el sistema de cronometraje.

Encapsula toda la lógica SQLite: creación del esquema, inserción de cruces
y consultas de resultados.  Thread-safe mediante check_same_thread=False +
un único objeto Connection compartido con WAL activado.
"""

import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Ruta por defecto — se puede sobreescribir al instanciar
_DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "carreras.db"
_SCHEMA_SQL = Path(__file__).resolve().parents[1] / "data" / "schema.sql"


class DBManager:
    """Gestiona la base de datos SQLite del cronómetro."""

    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,   # ByteTrack + captura corren en hilos distintos
        )
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("Base de datos lista: %s", self.db_path)

    # ──────────────────────────────────────────────────────────────────────────
    #  Inicialización
    # ──────────────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        """Aplica schema.sql si las tablas no existen."""
        if _SCHEMA_SQL.exists():
            self._conn.executescript(_SCHEMA_SQL.read_text())
        else:
            # Fallback: DDL inline por si falta el archivo .sql
            self._conn.executescript("""
                PRAGMA journal_mode = WAL;
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS corredores (
                    id_registro       INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id          INTEGER NOT NULL,
                    dorsal            TEXT    DEFAULT 'N/A',
                    tiempo_cruce      TEXT    NOT NULL,
                    tiempo_carrera_ms INTEGER,
                    con_casco         INTEGER NOT NULL CHECK (con_casco IN (0, 1)),
                    foto_meta_path    TEXT,
                    UNIQUE (track_id, tiempo_cruce)
                );
                CREATE TABLE IF NOT EXISTS configuracion (
                    clave TEXT PRIMARY KEY,
                    valor TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_track_id     ON corredores (track_id);
                CREATE INDEX IF NOT EXISTS idx_dorsal       ON corredores (dorsal);
                CREATE INDEX IF NOT EXISTS idx_tiempo_cruce ON corredores (tiempo_cruce);
                CREATE INDEX IF NOT EXISTS idx_con_casco    ON corredores (con_casco);
            """)
        self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Agrega columnas/tablas faltantes en bases de datos existentes."""
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(corredores)")}
        if "tiempo_carrera_ms" not in cols:
            self._conn.execute(
                "ALTER TABLE corredores ADD COLUMN tiempo_carrera_ms INTEGER"
            )
            self._conn.commit()
            logger.info("Migracion: columna tiempo_carrera_ms agregada")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS configuracion (
                clave TEXT PRIMARY KEY,
                valor TEXT NOT NULL
            )
        """)
        self._conn.commit()

    # ──────────────────────────────────────────────────────────────────────────
    #  Escritura
    # ──────────────────────────────────────────────────────────────────────────

    def registrar_cruce(
        self,
        track_id: int,
        con_casco: bool,
        dorsal: str = "N/A",
        foto_meta_path: Optional[str] = None,
        tiempo_cruce: Optional[datetime] = None,
        tiempo_carrera_ms: Optional[int] = None,
    ) -> Optional[int]:
        """
        Inserta un cruce de línea detectado.

        Args:
            track_id:          ID de ByteTrack.
            con_casco:         True si YOLO detectó casco en ese frame.
            dorsal:            Texto devuelto por OCR, o 'N/A'.
            foto_meta_path:    Ruta al JPEG capturado.
            tiempo_cruce:      Si es None usa datetime.now() con microsegundos.
            tiempo_carrera_ms: Milisegundos desde el inicio de carrera, o None.

        Returns:
            id_registro insertado, o None si era un duplicado.
        """
        ts = tiempo_cruce or datetime.now(tz=timezone.utc)
        ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}"

        try:
            cur = self._conn.execute(
                """
                INSERT INTO corredores
                    (track_id, dorsal, tiempo_cruce, tiempo_carrera_ms, con_casco, foto_meta_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (track_id, dorsal, ts_str, tiempo_carrera_ms, int(con_casco), foto_meta_path),
            )
            self._conn.commit()
            logger.debug(
                "Cruce registrado — id=%d track=%d dorsal=%s casco=%s carrera_ms=%s",
                cur.lastrowid, track_id, dorsal, con_casco, tiempo_carrera_ms,
            )
            return cur.lastrowid
        except sqlite3.IntegrityError:
            # UNIQUE (track_id, tiempo_cruce) — detección duplicada, se ignora
            logger.debug("Cruce duplicado ignorado: track_id=%d ts=%s", track_id, ts_str)
            return None

    # ──────────────────────────────────────────────────────────────────────────
    #  Consultas
    # ──────────────────────────────────────────────────────────────────────────

    def actualizar_dorsal(self, id_registro: int, dorsal: str) -> None:
        """Asigna o corrige el dorsal de un cruce ya registrado."""
        self._conn.execute(
            "UPDATE corredores SET dorsal = ? WHERE id_registro = ?",
            (dorsal, id_registro),
        )
        self._conn.commit()

    # ── Inicio de carrera ─────────────────────────────────────────────────────

    def guardar_inicio_carrera(self, ts_ns: int) -> None:
        """Persiste el timestamp de inicio de carrera en nanosegundos."""
        self._conn.execute(
            "INSERT OR REPLACE INTO configuracion (clave, valor) VALUES ('inicio_carrera_ns', ?)",
            (str(ts_ns),),
        )
        self._conn.commit()
        logger.info("Inicio de carrera guardado: ts_ns=%d", ts_ns)

    def get_inicio_carrera(self) -> Optional[int]:
        """Retorna el timestamp de inicio de carrera en ns, o None si no se ha iniciado."""
        row = self._conn.execute(
            "SELECT valor FROM configuracion WHERE clave = 'inicio_carrera_ns'"
        ).fetchone()
        return int(row["valor"]) if row else None

    def guardar_fin_carrera(self, ts_ns: int) -> None:
        """Persiste el timestamp de fin de carrera en nanosegundos."""
        self._conn.execute(
            "INSERT OR REPLACE INTO configuracion (clave, valor) VALUES ('fin_carrera_ns', ?)",
            (str(ts_ns),),
        )
        self._conn.commit()
        logger.info("Fin de carrera guardado: ts_ns=%d", ts_ns)

    def get_fin_carrera(self) -> Optional[int]:
        """Retorna el timestamp de fin de carrera en ns, o None si la carrera sigue activa."""
        row = self._conn.execute(
            "SELECT valor FROM configuracion WHERE clave = 'fin_carrera_ns'"
        ).fetchone()
        return int(row["valor"]) if row else None

    def exportar_csv(self, output_path: str | Path) -> int:
        """
        Exporta los resultados a CSV ordenados por categoría y tiempo.
        Retorna la cantidad de filas exportadas.
        """
        import csv
        rows = self._conn.execute("""
            SELECT
                c.id_registro,
                c.dorsal,
                c.tiempo_cruce,
                c.tiempo_carrera_ms,
                c.con_casco,
                c.foto_meta_path,
                COALESCE(a.nombre,    '—') AS nombre,
                COALESCE(a.categoria, '—') AS categoria,
                COALESCE(a.equipo,    '')  AS equipo
            FROM corredores c
            LEFT JOIN atletas a ON c.dorsal = a.dorsal
            ORDER BY a.categoria, COALESCE(c.tiempo_carrera_ms, 999999999), c.tiempo_cruce
        """).fetchall()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def _fmt_tiempo(ms):
            if ms is None:
                return ""
            total_s  = ms // 1000
            resto_ms = ms % 1000
            return f"{total_s // 60:02d}:{total_s % 60:02d}.{resto_ms:03d}"

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "Posicion", "Dorsal", "Nombre", "Categoria", "Equipo",
                "Tiempo_carrera", "Tiempo_cruce", "Con_casco", "Foto"
            ])
            for pos, r in enumerate(rows, 1):
                w.writerow([
                    pos,
                    r["dorsal"],
                    r["nombre"],
                    r["categoria"],
                    r["equipo"],
                    _fmt_tiempo(r["tiempo_carrera_ms"]),
                    r["tiempo_cruce"],
                    "Si" if r["con_casco"] else "No",
                    r["foto_meta_path"] or "",
                ])

        logger.info("CSV exportado: %s  (%d filas)", output_path, len(rows))
        return len(rows)

    # ── Atletas ───────────────────────────────────────────────────────────────

    def registrar_atleta(self, dorsal: str, nombre: str,
                         categoria: str, equipo: str = "") -> None:
        """Inserta o reemplaza un atleta pre-registrado."""
        self._conn.execute(
            "INSERT OR REPLACE INTO atletas (dorsal, nombre, categoria, equipo) VALUES (?,?,?,?)",
            (str(dorsal), nombre, categoria, equipo),
        )
        self._conn.commit()

    def buscar_atleta(self, dorsal: str) -> Optional[sqlite3.Row]:
        """Devuelve el atleta con ese dorsal, o None si no existe."""
        return self._conn.execute(
            "SELECT * FROM atletas WHERE dorsal = ?", (str(dorsal),)
        ).fetchone()

    def get_todos_atletas(self) -> list:
        return self._conn.execute(
            "SELECT * FROM atletas ORDER BY categoria, CAST(dorsal AS INTEGER)"
        ).fetchall()

    def get_podio(self) -> list:
        """
        Retorna los cruces con dorsal asignado, ordenados por categoría y tiempo.
        Hace JOIN con atletas para traer nombre y categoría.
        """
        return self._conn.execute("""
            SELECT
                c.id_registro,
                c.dorsal,
                c.tiempo_cruce,
                c.tiempo_carrera_ms,
                c.con_casco,
                c.foto_meta_path,
                COALESCE(a.nombre,    '—') AS nombre,
                COALESCE(a.categoria, '—') AS categoria,
                COALESCE(a.equipo,    '')  AS equipo
            FROM corredores c
            LEFT JOIN atletas a ON c.dorsal = a.dorsal
            WHERE c.dorsal != 'N/A' AND c.dorsal IS NOT NULL
            ORDER BY a.categoria, COALESCE(c.tiempo_carrera_ms, 999999999), c.tiempo_cruce
        """).fetchall()

    def obtener_cruces(
        self,
        solo_con_casco: Optional[bool] = None,
        dorsal: Optional[str] = None,
        limite: int = 500,
        limit: int = 500,
    ) -> list[sqlite3.Row]:
        """
        Devuelve registros filtrados, ordenados por tiempo_cruce ASC.

        Args:
            solo_con_casco: True → solo con casco | False → solo sin casco | None → todos.
            dorsal:         Filtrar por número de dorsal específico.
            limit:          Máximo de filas devueltas.
        """
        filters, params = [], []

        if solo_con_casco is not None:
            filters.append("con_casco = ?")
            params.append(int(solo_con_casco))
        if dorsal is not None:
            filters.append("dorsal = ?")
            params.append(dorsal)

        where = ("WHERE " + " AND ".join(filters)) if filters else ""
        params.append(limite or limit)

        rows = self._conn.execute(
            f"SELECT * FROM corredores {where} ORDER BY tiempo_cruce ASC LIMIT ?",
            params,
        ).fetchall()
        return rows

    def total_cruces(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM corredores").fetchone()[0]

    def sin_casco(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM corredores WHERE con_casco = 0"
        ).fetchone()[0]

    # ──────────────────────────────────────────────────────────────────────────
    #  Ciclo de vida
    # ──────────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
