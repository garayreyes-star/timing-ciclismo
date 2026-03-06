"""
judge_server.py — Interfaz web para que el juez asigne dorsales.

Corre en un hilo daemon en http://0.0.0.0:8080
El juez abre desde tablet/celular en la misma red:  http://<IP>:8080
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template_string, request

logger = logging.getLogger("judge_server")

# ─── HTML de la interfaz ──────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cronometraje — Juez</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: Arial, sans-serif; background: #111; color: #eee; padding: 10px; }
  h1 { text-align: center; color: #f90; padding: 10px 0 20px; font-size: 1.4rem; }
  .cruce {
    background: #1e1e1e; border: 1px solid #333; border-radius: 10px;
    margin-bottom: 16px; padding: 12px; display: flex; gap: 14px; align-items: center;
  }
  .cruce img { width: 120px; height: 120px; object-fit: cover; border-radius: 8px; border: 2px solid #555; }
  .cruce .info { flex: 1; }
  .cruce .tiempo { font-size: 1.5rem; font-weight: bold; color: #0f0; font-family: monospace; }
  .cruce .tid { color: #aaa; font-size: 0.85rem; margin: 4px 0 10px; }
  .cruce .casco { font-size: 0.9rem; margin-bottom: 8px; }
  .cruce input {
    width: 100%; padding: 10px; font-size: 1.4rem; border-radius: 6px;
    border: 2px solid #555; background: #222; color: #fff; text-align: center;
    font-weight: bold; letter-spacing: 3px;
  }
  .cruce input:focus { border-color: #f90; outline: none; }
  .cruce .saved { color: #0f0; font-size: 0.85rem; margin-top: 4px; display: none; }
  .no-foto { width: 120px; height: 120px; background: #333; border-radius: 8px;
    display:flex; align-items:center; justify-content:center; color:#666; font-size:0.8rem; text-align:center; }
  .header-bar { display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; }
  .btn-refresh { background:#333; color:#eee; border:none; padding:8px 16px; border-radius:6px; cursor:pointer; }
</style>
</head>
<body>
<h1>🚴 Cronometraje — Panel del Juez</h1>
<div class="header-bar">
  <span id="total" style="color:#aaa;font-size:0.9rem;"></span>
  <button class="btn-refresh" onclick="cargar()">↺ Actualizar</button>
</div>
<div id="lista"></div>

<script>
async function cargar() {
  const res = await fetch('/api/cruces');
  const data = await res.json();
  document.getElementById('total').textContent = data.length + ' cruce(s) registrado(s)';
  const lista = document.getElementById('lista');
  lista.innerHTML = '';
  data.reverse().forEach(c => {
    const div = document.createElement('div');
    div.className = 'cruce';
    const foto = c.foto
      ? `<img src="/foto/${encodeURIComponent(c.foto)}" onerror="this.parentNode.innerHTML='<div class=no-foto>Sin foto</div>'">`
      : '<div class="no-foto">Sin foto</div>';
    div.innerHTML = `
      ${foto}
      <div class="info">
        <div class="tiempo">${c.tiempo}</div>
        <div class="tid">Track ID: ${c.track_id}  |  ID: #${c.id}</div>
        <div class="casco">${c.con_casco ? '✅ Con casco' : '❌ Sin casco'}</div>
        <input type="text" inputmode="numeric" pattern="[0-9]*"
               placeholder="Dorsal..." value="${c.dorsal !== 'N/A' ? c.dorsal : ''}"
               onchange="guardar(${c.id}, this)"
               onkeydown="if(event.key==='Enter') guardar(${c.id}, this)">
        <div class="saved" id="ok_${c.id}">✓ Guardado</div>
      </div>`;
    lista.appendChild(div);
  });
}

async function guardar(id, input) {
  const dorsal = input.value.trim();
  if (!dorsal) return;
  await fetch('/api/dorsal', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({id, dorsal})
  });
  const ok = document.getElementById('ok_' + id);
  ok.style.display = 'block';
  setTimeout(() => ok.style.display = 'none', 2000);
}

cargar();
setInterval(cargar, 5000);  // auto-actualizar cada 5 s
</script>
</body>
</html>"""


# ─── Servidor ─────────────────────────────────────────────────────────────────

class JudgeServer:
    """
    Servidor Flask liviano para asignación de dorsales por el juez.

    Uso::
        server = JudgeServer(db)
        server.start()   # arranca en hilo daemon, no bloquea
    """

    def __init__(self, db, captures_dir: str = "data/captures", port: int = 8080) -> None:
        self._db   = db
        self._dir  = Path(captures_dir)
        self._port = port
        self._app  = self._build_app()

    def _build_app(self) -> Flask:
        app = Flask(__name__)
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)   # silenciar logs de requests

        @app.route("/")
        def index():
            return render_template_string(_HTML)

        @app.route("/api/cruces")
        def cruces():
            rows = self._db.obtener_cruces(limite=50)
            result = []
            for r in rows:
                result.append({
                    "id":        r["id_registro"],
                    "track_id":  r["track_id"],
                    "dorsal":    r["dorsal"] or "N/A",
                    "tiempo":    r["tiempo_cruce"],
                    "con_casco": bool(r["con_casco"]),
                    "foto":      r["foto_meta_path"],
                })
            return jsonify(result)

        @app.route("/foto/<path:foto_path>")
        def foto(foto_path):
            from flask import send_file
            p = Path(foto_path)
            if not p.exists():
                p = self._dir / foto_path
            if p.exists():
                return send_file(str(p), mimetype="image/jpeg")
            return "Not found", 404

        @app.route("/api/dorsal", methods=["POST"])
        def set_dorsal():
            data   = request.get_json()
            id_reg = int(data["id"])
            dorsal = str(data["dorsal"]).strip()[:10]
            self._db.actualizar_dorsal(id_reg, dorsal)
            logger.info("Dorsal actualizado: id=%d dorsal=%s", id_reg, dorsal)
            return jsonify({"ok": True})

        return app

    def start(self) -> None:
        t = threading.Thread(
            target=lambda: self._app.run(host="0.0.0.0", port=self._port, use_reloader=False),
            name="JudgeServer",
            daemon=True,
        )
        t.start()
        logger.info("Panel del juez en http://0.0.0.0:%d", self._port)
