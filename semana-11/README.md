# Semana 11 — Modelos multimodales (imagen ↔ texto)

Esta carpeta contiene dos notebooks sobre la relación entre **imágenes** y **texto**:

| Notebook | Tema | Qué hace | Stack |
| --- | --- | --- | --- |
| [`Tema-21/Tema_21.ipynb`](Tema-21/Tema_21.ipynb) | **Imagen → texto** (*image captioning*) | Genera una descripción en lenguaje natural de una imagen usando **BLIP**. | `transformers`, `torch`, `pillow` |
| [`Tema-22/Tema_22.ipynb`](Tema-22/Tema_22.ipynb) | **Mini-CLIP** (imagen ↔ texto) | Entrena un modelo **contrastivo** (encoder de imagen CNN + encoder de texto por *embeddings*) sobre figuras de colores sintéticas y hace **búsqueda texto→imagen** con `Recall@K`. | `tensorflow`, `numpy`, `matplotlib` |

Ambos notebooks pueden ejecutarse **localmente** (VS Code / Jupyter) o en **Google Colab** (badge *Open In Colab* en la primera celda de cada uno).

---

## ¿Se pueden correr en Google Colab?

**Sí, ambos.**

- **Tema 21 (BLIP):** compatible con Colab. La celda de *setup* instala `transformers`, `torch` y `pillow` solo si detecta Colab. Si no colocas una imagen propia, el notebook **descarga una imagen de ejemplo** automáticamente. Recomendado usar **GPU** (`Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU`).
- **Tema 22 (mini-CLIP):** compatible con Colab. Usa `tensorflow`, `numpy` y `matplotlib`, ya preinstalados en Colab; el dataset es **sintético** (no requiere archivos externos). Funciona en CPU y es más rápido con GPU.

---

## Requisitos locales

- **Python 3.10 o 3.11** (recomendado por compatibilidad con TensorFlow y PyTorch).
- ~3–5 GB libres en disco (la primera ejecución de BLIP descarga el modelo; TensorFlow ocupa espacio).
- **GPU opcional** pero recomendada para BLIP.

---

## Opción 1 — Conda / Miniconda

1. Instala [Miniconda](https://docs.conda.io/en/latest/miniconda.html) o [Anaconda](https://www.anaconda.com/download) si aún no lo tienes.

2. Abre una terminal en esta carpeta (`semana-11`) y crea el entorno con Python 3.11:

   ```bash
   conda create -n semana11 python=3.11 -y
   conda activate semana11
   ```

3. Instala las dependencias con `pip` dentro del entorno:

   ```bash
   pip install --upgrade pip
   # Tema 21 (BLIP)
   pip install transformers torch pillow
   # Tema 22 (mini-CLIP)
   pip install tensorflow numpy matplotlib
   ```

4. Registra el entorno como kernel de Jupyter (para verlo en VS Code / Jupyter):

   ```bash
   pip install ipykernel
   python -m ipykernel install --user --name semana11 --display-name "Python (semana11)"
   ```

Para salir del entorno cuando termines: `conda deactivate`.

---

## Opción 2 — venv (Python estándar)

1. Abre una terminal en esta carpeta (`semana-11`).

2. Crea y activa el entorno virtual:

   **macOS / Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows (PowerShell)**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Instala las dependencias:

   ```bash
   pip install --upgrade pip
   pip install transformers torch pillow tensorflow numpy matplotlib ipykernel
   ```

4. (Opcional) Registra el kernel para que aparezca por nombre en VS Code / Jupyter:

   ```bash
   python -m ipykernel install --user --name semana11-venv --display-name "Python (semana11-venv)"
   ```

Para salir del entorno: `deactivate`.

---

## Ejecutar los notebooks

### VS Code (recomendado)

1. Abre la carpeta del proyecto en VS Code e instala la extensión **Jupyter**.
2. Abre `Tema-21/Tema_21.ipynb` o `Tema-22/Tema_22.ipynb`.
3. Arriba a la derecha, selecciona el kernel del entorno que creaste (`Python (semana11)` o el intérprete de `.venv`).
4. Ejecuta las celdas con `Shift + Enter`. La primera celda de instalación (`!pip install ...`) solo actúa en Colab; en local puedes saltarla porque ya instalaste las dependencias.

### Jupyter en el navegador

```bash
jupyter notebook Tema-21/Tema_21.ipynb
```

---

## Notas y solución de problemas

- **Tema 21 — imagen de entrada:** el notebook busca `sample.jpg` junto al notebook; si no existe, descarga una imagen de ejemplo. Puedes cambiar `ruta_imagen` para usar tu propia imagen.
- **Tema 21 — primera ejecución lenta:** la primera llamada descarga los pesos de BLIP (varios cientos de MB) desde Hugging Face. Es normal que tarde.
- **Apple Silicon (M1/M2/M3):**
  - PyTorch: `pip install torch` ya funciona (con aceleración **MPS**).
  - TensorFlow: si quieres acelerar con la GPU (Metal), añade `pip install tensorflow-metal`.
- **GPU NVIDIA (Linux/Windows):** sigue https://www.tensorflow.org/install/pip (TensorFlow con CUDA) y https://pytorch.org/get-started/locally/ (PyTorch con CUDA).
- **Tema 22 — reproducibilidad:** se fija `SEED = 42`; con pocas épocas el `Recall@K` puede variar. Sube `EPOCHS` si quieres mejores resultados.
- **Conflictos entre TensorFlow y PyTorch:** conviven bien en el mismo entorno, pero si aparecen conflictos raros, crea **dos entornos separados** (uno para Tema 21 y otro para Tema 22).
