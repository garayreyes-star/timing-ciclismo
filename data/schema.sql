-- =============================================================================
--  schema.sql — Sistema de cronometraje para corredores
--  Orange Pi 6 Plus · CIX P1 · YOLO + ByteTrack + OCR + Mali-G610
-- =============================================================================

PRAGMA journal_mode = WAL;       -- escrituras concurrentes sin bloquear lecturas
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
--  Tabla principal: un registro por cruce de línea detectado
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS corredores (
    id_registro     INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id        INTEGER NOT NULL,       -- ID asignado por ByteTrack
    dorsal          TEXT    DEFAULT 'N/A',  -- Número detectado por OCR (opcional)
    tiempo_cruce    TEXT    NOT NULL,       -- ISO-8601 con milisegundos: "2025-06-01T10:23:45.123"
    con_casco       INTEGER NOT NULL        -- 1 = casco detectado, 0 = sin casco  (BOOLEAN en SQLite)
                    CHECK (con_casco IN (0, 1)),
    foto_meta_path  TEXT,                   -- Ruta a la captura guardada por Mali-G610

    -- Evitar duplicados: el mismo track_id no puede cruzar dos veces
    -- en menos de 2 segundos (protección contra detecciones duplicadas)
    UNIQUE (track_id, tiempo_cruce)
);

-- ---------------------------------------------------------------------------
--  Atletas pre-registrados: dorsal → corredor
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS atletas (
    dorsal      TEXT    PRIMARY KEY,          -- "42", "007", etc.
    nombre      TEXT    NOT NULL,
    categoria   TEXT    NOT NULL DEFAULT '',  -- "Elite", "Sub-23", "Femenino", etc.
    equipo      TEXT    DEFAULT ''
);

-- ---------------------------------------------------------------------------
--  Índices para consultas frecuentes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_track_id     ON corredores (track_id);
CREATE INDEX IF NOT EXISTS idx_dorsal       ON corredores (dorsal);
CREATE INDEX IF NOT EXISTS idx_tiempo_cruce ON corredores (tiempo_cruce);
CREATE INDEX IF NOT EXISTS idx_con_casco    ON corredores (con_casco);
