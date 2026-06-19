# Tema 12 — Clasificación de imágenes con CNN y Transfer Learning

Guía para ejecutar **`Tema_12.ipynb`** de forma **local**. En Google Colab el entrenamiento puede tardar mucho (sobre todo el *transfer learning* con EfficientNetB0 a 224×224), por lo que correrlo en tu equipo —idealmente con **GPU**— suele ser más rápido y cómodo.

---

## 1. Requisitos

- **Python 3.10 – 3.12**
- ~5 GB de espacio libre (dataset CIFAR-10 + pesos de EfficientNetB0)
- (Opcional pero recomendado) **GPU**:
  - **NVIDIA** (Linux/Windows): TensorFlow con CUDA.
  - **Apple Silicon** (M1/M2/M3): usar `tensorflow-macos` + `tensorflow-metal`.

> Sin GPU el notebook funciona igual, pero el entrenamiento es bastante más lento.

---

## 2. Crear un entorno

Puedes usar **venv** (incluido en Python) o **Conda**. Elige una de las dos opciones.

### Opción A — venv

**macOS / Linux**
```bash
cd "semana-06/Tema-12"
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
cd "semana-06\Tema-12"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Opción B — Conda

```bash
cd "semana-06/Tema-12"
conda create -n tema12 python=3.11 -y
conda activate tema12
```

> Si prefieres instalar todo con Conda (incluido TensorFlow) en lugar de `pip`:
> ```bash
> conda install -c conda-forge tensorflow matplotlib numpy scikit-learn seaborn jupyter -y
> ```
> En ese caso puedes saltarte el paso 3. Para desactivar el entorno al terminar: `conda deactivate`.

---

## 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install tensorflow matplotlib numpy scikit-learn seaborn jupyter
```

### Apple Silicon (M1/M2/M3)
En vez de `tensorflow`, instala las versiones con aceleración Metal:
```bash
pip install tensorflow-macos tensorflow-metal matplotlib numpy scikit-learn seaborn jupyter
```

### NVIDIA con GPU (Linux)
```bash
pip install "tensorflow[and-cuda]" matplotlib numpy scikit-learn seaborn jupyter
```

---

## 4. Ejecutar el notebook

**Opción A — VS Code**
1. Abre la carpeta del proyecto en VS Code.
2. Abre `Tema_12.ipynb`.
3. Selecciona el kernel de tu entorno (arriba a la derecha): el `.venv` o el entorno de Conda (`tema12`), según la opción que elegiste en el paso 2.
4. Ejecuta las celdas en orden con **Run All**.

**Opción B — Jupyter en el navegador**
```bash
jupyter notebook Tema_12.ipynb
```

---

## 5. Verificar que la GPU está activa (opcional)

Ejecuta en una celda o terminal:
```python
import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices("GPU"))
```
Si la lista está vacía, el entrenamiento usará CPU.

---

## 6. Recomendaciones para acelerar / probar rápido

El notebook entrena con CIFAR-10 completo. Si solo quieres **validar que todo corre** sin esperar tanto, puedes reducir temporalmente la carga:

- **Menos épocas**: baja `epochs=15` (CNN) y `epochs=8` (transfer learning) a `epochs=2`.
- **Submuestrear el dataset** justo después de cargarlo:
  ```python
  x_train, y_train = x_train[:5000], y_train[:5000]
  x_test,  y_test  = x_test[:1000],  y_test[:1000]
  ```
- **Cuidado con la memoria**: la celda de transfer learning redimensiona las imágenes a 224×224. Si te quedas sin RAM, reduce el dataset como arriba o procesa por lotes.

---

## 7. Problemas comunes

| Problema | Solución |
|----------|----------|
| `No module named tensorflow` | Activa el entorno (venv o Conda) e instala las dependencias (paso 3). |
| Entrenamiento muy lento | Verifica la GPU (paso 5) o reduce épocas/dataset (paso 6). |
| `ResourceExhaustedError` / sin memoria | Baja el `batch_size`, reduce el dataset o cierra otras apps. |
| La descarga de CIFAR-10 falla | Reintenta; se cachea en `~/.keras/datasets/`. |
| Pesos de EfficientNet no descargan | Revisa tu conexión; se cachean en `~/.keras/models/`. |

---

## 8. Estructura

```
Tema-12/
├── Tema_12.ipynb   # Notebook principal
├── tema_12.py      # Versión exportada en script
└── README.md       # Esta guía
```
