# Actividad 8 — Generación de imágenes texto→imagen (Stable Diffusion + KerasCV)

Guía para ejecutar **`Actividad_8.ipynb`** de forma **local** (VS Code / Jupyter). También puedes abrirlo en **Google Colab** con el badge del propio notebook, pero aquí se explica cómo montar el entorno en tu máquina con **conda** o con **venv**.

## Requisitos

- **Python 3.10 o 3.11** (recomendado para compatibilidad con TensorFlow / KerasCV).
- ~5–10 GB libres en disco (la primera generación descarga los pesos de Stable Diffusion, varios GB).
- **GPU opcional pero muy recomendada**: en CPU la generación funciona, pero es lenta.

## Dependencias

| Paquete | Uso |
| --- | --- |
| `tensorflow` | Backend del modelo. |
| `keras-cv` | Modelo Stable Diffusion (`text_to_image`). |
| `matplotlib` | Visualizar las imágenes generadas. |
| `numpy` | Manejo de arreglos de imágenes. |
| `transformers`, `torch` | **Solo para el reto opcional** (CLIP / BLIP). |

---

## Opción 1 — Conda / Miniconda

1. Instala [Miniconda](https://docs.conda.io/en/latest/miniconda.html) o [Anaconda](https://www.anaconda.com/download) si aún no lo tienes.

2. Abre una terminal en esta carpeta (`semana-10`) y crea el entorno con Python 3.11:

   ```bash
   conda create -n actividad8 python=3.11 -y
   conda activate actividad8
   ```

3. Instala las dependencias con `pip` dentro del entorno:

   ```bash
   pip install --upgrade pip
   pip install tensorflow keras-cv matplotlib numpy
   # Reto opcional (CLIP / BLIP):
   # pip install transformers torch
   ```

4. Registra el entorno como kernel de Jupyter (para verlo en VS Code / Jupyter):

   ```bash
   python -m ipykernel install --user --name actividad8 --display-name "Python (actividad8)"
   ```

Para salir del entorno cuando termines: `conda deactivate`.

---

## Opción 2 — venv (Python estándar)

1. Abre una terminal en esta carpeta (`semana-10`).

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
   pip install tensorflow keras-cv matplotlib numpy ipykernel
   # Reto opcional (CLIP / BLIP):
   # pip install transformers torch
   ```

4. (Opcional) Registra el kernel para que aparezca por nombre en VS Code / Jupyter:

   ```bash
   python -m ipykernel install --user --name actividad8-venv --display-name "Python (actividad8-venv)"
   ```

Para salir del entorno: `deactivate`.

---

## Ejecutar el notebook

### VS Code (recomendado)

1. Abre la carpeta del proyecto en VS Code e instala la extensión **Jupyter**.
2. Abre `semana-10/Actividad_8.ipynb`.
3. Arriba a la derecha, selecciona el kernel del entorno que creaste:
   - Conda: `Python (actividad8)`.
   - venv: el intérprete dentro de `.venv` o `Python (actividad8-venv)`.
4. Ejecuta las celdas con `Shift + Enter`. La celda de instalación (`!pip install ...`) solo actúa en Colab; en local puedes saltarla porque ya instalaste las dependencias.

### Jupyter en el navegador

```bash
jupyter notebook Actividad_8.ipynb
```

---

## Notas y solución de problemas

- **Primera generación lenta**: la primera llamada a `text_to_image` descarga los pesos de Stable Diffusion (~GB) y compila el modelo. Es normal que tarde varios minutos.
- **Compatibilidad con Keras 3**: si aparecen errores al cargar `keras_cv.models.StableDiffusion`, fija versiones compatibles y reinicia el kernel:

  ```bash
  pip install "keras-cv==0.9.0" "tensorflow==2.15.*"
  ```

- **GPU NVIDIA (Linux/Windows)**: instala la build de TensorFlow con soporte CUDA siguiendo https://www.tensorflow.org/install/pip. Con GPU puedes activar precisión mixta descomentando en el notebook:
  `tf.keras.mixed_precision.set_global_policy("mixed_float16")`.
- **Apple Silicon (M1/M2/M3)**: `pip install tensorflow` ya funciona en CPU. Para aceleración con la GPU integrada (Metal), añade:

  ```bash
  pip install tensorflow-metal
  ```

  Si tienes conflictos, alternativa: `pip install tensorflow-macos tensorflow-metal`.
- **CPU sin GPU**: funciona igual; reduce `batch_size` en `text_to_image` a 1 para ahorrar memoria si es necesario.
