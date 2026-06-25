# Semana 07 — Reconocimiento facial y seguimiento de objetos

Guía para ejecutar **localmente** los notebooks de esta semana:

- **`Tema-13/Tema_13.ipynb`** — Reconocimiento facial y verificación biométrica (MTCNN + FaceNet + distancia coseno).
- **`Tema-14/Tema_14.ipynb`** — Seguimiento de objetos (*tracking*) con YOLO y BoT-SORT.

> Ambos notebooks también traen un badge **Open In Colab** y una celda **Setup para Google Colab**. Esta guía es para correrlos en tu propio equipo (recomendado para Tema 14, que usa una ventana en vivo con `cv2.imshow`).

---

## 1. Requisitos

- **Python 3.10 – 3.12**
- ~5 GB de espacio libre (modelos de FaceNet, YOLO y dependencias de TensorFlow/PyTorch)
- (Opcional pero recomendado) **GPU**:
  - **NVIDIA** (Linux/Windows): PyTorch con CUDA acelera el tracking de Tema 14.
  - **Apple Silicon** (M1/M2/M3): PyTorch usa el backend `mps` automáticamente.

> Sin GPU los notebooks funcionan igual, pero la inferencia es más lenta.

---

## 2. Crear un entorno

Puedes usar **Conda** (recomendada) o **venv**. Elige una sola opción.

### Opción A — Conda / Miniconda

```bash
cd "semana-07"
conda create -n semana07 python=3.11 -y
conda activate semana07
```

Para registrar el entorno como kernel de Jupyter (para que aparezca en VS Code y Jupyter):

```bash
python -m ipykernel install --user --name semana07 --display-name "Python (semana07)"
```

Para salir del entorno al terminar: `conda deactivate`.

### Opción B — venv (Python estándar)

**macOS / Linux**
```bash
cd "semana-07"
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
cd "semana-07"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## 3. Instalar dependencias

Con el entorno **activado**, actualiza `pip` e instala los paquetes:

```bash
pip install --upgrade pip
pip install jupyter ipykernel numpy matplotlib scipy opencv-python mtcnn keras-facenet ultralytics
```

¿Para qué sirve cada paquete?

| Paquete | Notebook | Uso |
| --- | --- | --- |
| `opencv-python` | Tema 13 y 14 | Lectura/escritura de imágenes y video, dibujo de cajas. |
| `mtcnn` | Tema 13 | Detección de rostros (*bounding boxes*). |
| `keras-facenet` | Tema 13 | Embeddings biométricos de 512 dimensiones (trae **TensorFlow** como dependencia). |
| `scipy` | Tema 13 | Distancia coseno para la verificación. |
| `ultralytics` | Tema 14 | Modelo **YOLO** + trackers (**BoT-SORT**); instala **PyTorch** automáticamente. |
| `numpy`, `matplotlib` | Tema 13 y 14 | Cálculo numérico y visualización. |
| `jupyter`, `ipykernel` | Ambos | Ejecutar los notebooks. |

> 💡 **Apple Silicon (M1/M2/M3):** si TensorFlow no aprovecha la GPU, instala además `tensorflow-metal`:
> ```bash
> pip install tensorflow-metal
> ```

---

## 4. Archivos de datos necesarios

Estos archivos **no** se incluyen en el repo; consíguelos antes de ejecutar:

- **Tema 13:** si no subes tus propias imágenes, el notebook **descarga automáticamente** fotos de ejemplo (`foto_pasaporte.png`, `foto_selfie.png`, `otra_persona.png`). Para usar las tuyas, colócalas en `Tema-13/` con esos nombres.
- **Tema 14:** necesitas el video de prueba **`MOT17 04 FRCNN raw.mp4`** dentro de `Tema-14/`. El modelo `yolo26x.pt` lo descarga `ultralytics` la primera vez.

---

## 5. Ejecutar los notebooks

### VS Code (recomendado)

1. Abre la carpeta `02-Codigo` en VS Code.
2. Instala la extensión **Jupyter** si no la tienes.
3. Abre `semana-07/Tema-13/Tema_13.ipynb` o `semana-07/Tema-14/Tema_14.ipynb`.
4. En la esquina superior derecha selecciona el kernel:
   - **Conda:** `Python (semana07)`.
   - **venv:** el intérprete dentro de `.venv`.
5. Ejecuta las celdas con `Shift + Enter`.

### Jupyter en el navegador

```bash
jupyter notebook
```

Luego abre el notebook que quieras desde el navegador.

---

## 6. Notas

- En **Tema 14**, el código **detecta el entorno automáticamente**: en local muestra el seguimiento en vivo con `cv2.imshow` (sal con la tecla **`q`**) y siempre guarda el resultado en `salida_tracking.mp4`, que se reproduce en la última celda.
- Si `cv2.imshow` da problemas en Linux, instala las dependencias del sistema (por ejemplo `sudo apt install libgl1`) o usa la celda final para ver el video resultante.
