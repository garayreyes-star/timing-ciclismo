#!/bin/bash
# start.sh — Lanzador del sistema de cronometraje para PC Linux x86_64
#
# Uso:
#   bash start.sh                         # camara 0, display activado
#   bash start.sh --camera 1              # camara 1
#   bash start.sh --camera rtsp://...     # camara IP via RTSP
#   bash start.sh --no-display            # modo headless (sin ventana)
#   bash start.sh --linea 540 --conf 0.35 # calibrar linea de meta y confianza
#
# Instalar servicio systemd (arranque automatico):
#   sudo bash start.sh --install

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/venv/bin/python"
SERVICE_NAME="timing"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# ─── Instalar servicio systemd ─────────────────────────────────────────────────
if [[ "$1" == "--install" ]]; then
  echo "Instalando servicio systemd: ${SERVICE_FILE}"
  cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Cronometraje Ciclismo — CUDA YOLO + ByteTrack
After=network.target graphical.target
Wants=graphical.target

[Service]
Type=simple
User=${USER}
Environment="DISPLAY=:0"
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${VENV_PYTHON} ${SCRIPT_DIR}/main_timing.py --camera 0
Restart=on-failure
RestartSec=5

[Install]
WantedBy=graphical.target
EOF
  systemctl daemon-reload
  systemctl enable "${SERVICE_NAME}.service"
  systemctl start  "${SERVICE_NAME}.service"
  echo "Servicio instalado y arrancado."
  echo "  Estado:   systemctl status ${SERVICE_NAME}"
  echo "  Logs:     journalctl -u ${SERVICE_NAME} -f"
  echo "  Detener:  systemctl stop ${SERVICE_NAME}"
  exit 0
fi

# ─── Arranque normal ───────────────────────────────────────────────────────────

# Si existe venv local, usarlo; sino usar python del sistema
if [[ -f "$VENV_PYTHON" ]]; then
  PYTHON="$VENV_PYTHON"
else
  PYTHON="python3"
fi

exec "$PYTHON" "${SCRIPT_DIR}/main_timing.py" "$@"
