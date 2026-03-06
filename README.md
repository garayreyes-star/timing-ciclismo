# Cronometraje de Ciclismo — PC Linux x86_64 + CUDA

Port del sistema Orange Pi (NPU Zhouyi + Mali OpenCL) a PC Linux con GPU NVIDIA.

## Requisitos de sistema

- Linux x86_64 (Ubuntu 20.04+ / Debian 11+)
- Python 3.10+
- NVIDIA GTX 1650 (o superior) con drivers + CUDA 11.8 o 12.x
- Tkinter para el popup de dorsales: `sudo apt install python3-tk`

## Instalacion

```bash
# 1. Crear entorno virtual
cd timing-pc/
python3 -m venv venv
source venv/bin/activate

# 2. Instalar dependencias (reemplazar onnxruntime por onnxruntime-gpu para CUDA)
pip install -r requirements.txt
pip install onnxruntime-gpu   # GPU CUDA
# o: pip install onnxruntime  # CPU fallback

# 3. Obtener modelo YOLO
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640, opset=12)"
cp yolov8n.onnx models/
# o descargar directamente: https://github.com/ultralytics/assets/releases
```

## Uso

```bash
# Arrancar con camara USB indice 0, display activado
bash start.sh

# Camara USB indice 1
bash start.sh --camera 1

# Camara IP via RTSP (sin ventana)
bash start.sh --camera rtsp://192.168.1.10/stream --no-display

# Calibrar linea de meta (Y en pixeles segun tu resolucion y posicion de camara)
bash start.sh --linea 400

# Ajustar sensibilidad de deteccion
bash start.sh --conf 0.40

# Administracion de atletas (importar Excel, ver podio)
python3 admin.py
```

## Calibracion de la linea de meta

El parametro `--linea Y` define la coordenada Y (pixeles desde arriba) donde
los ciclistas cruzan. Depende de la posicion fisica de la camara:

- Con display activado (`--display`), observar el video y ajustar hasta que
  la linea roja coincida con la linea de meta fisica.
- Ejemplo: para 1080p con camara alta, la linea suele estar entre Y=400 y Y=700.

## Arquitectura

```
main_timing.py          Punto de entrada — TimingPipeline
detector.py             CUDAEngine = YOLODetector + ByteTrackReID
utils/
  crossing.py           DetectorCruce — algoritmo de cruce con histeresis
  db_manager.py         SQLite WAL — tabla corredores + atletas
  crop_worker.py        CropWorker — recorte JPEG en CPU (hilo separado)
  popup_cruce.py        PopupCruce — Tkinter: foto + entrada de dorsal + podio
  judge_server.py       JudgeServer — Flask :8080 — panel web para el juez
admin.py                Administracion de atletas (Tkinter, importa Excel/CSV)
data/schema.sql         Esquema SQLite
models/                 Modelos ONNX (no versionados)
```

## Panel del juez

Al arrancar, el sistema expone `http://<IP>:8080` — el juez puede abrir
desde cualquier dispositivo en la misma red para ver cruces en tiempo real
y asignar dorsales manualmente.

## Diferencias respecto al sistema Orange Pi

| Componente         | Orange Pi                      | PC Linux                     |
|--------------------|--------------------------------|------------------------------|
| Inferencia         | NPU Zhouyi (NOE_Engine, .cix)  | CUDA GPU (onnxruntime-gpu)   |
| Tracker            | SimpleTracker (IOU greedy)     | ByteTrack + ReID HSV cosine  |
| Crop worker        | MaliCropWorker (OpenCL)        | CropWorker (CPU)             |
| Camara             | /dev/video3 CAP_V4L2           | /dev/video0 o RTSP           |
| Deteccion de casco | Clase custom en yolo26.onnx    | No disponible con COCO       |
