# Tema 18 — Clasificación de imágenes *zero-shot* con CLIP

Instrucciones para crear el entorno y ejecutar **localmente** el notebook [`Tema_18.ipynb`](Tema_18.ipynb).

> La explicación de **lo que hace el código** está dentro del propio notebook, en la celda de markdown *"¿Qué hace este código?"*.

---

## 1. Requisitos

- **Python 3.10 – 3.12**
- ~2 GB de espacio libre (PyTorch + los pesos del modelo `clip-vit-base-patch32`, que se descargan la primera vez desde Hugging Face, ~600 MB).
- Conexión a internet la **primera** vez (para descargar el modelo y la imagen de ejemplo; el modelo después queda en caché en `~/.cache/huggingface`).
- (Opcional) Una imagen propia si no quieres usar la de ejemplo (ver [sección 4](#4-imagen-de-entrada)).
- (Opcional) **GPU NVIDIA** con CUDA para acelerar la inferencia. Sin GPU el notebook funciona igual, en CPU.

---

## 2. Crear un entorno

Puedes usar **Conda** (recomendada) o **venv**. Elige una sola opción.

### Opción A — Conda / Miniconda

```bash
cd "semana-09/Tema-18"
conda create -n tema18 python=3.11 -y
conda activate tema18
```

Para registrar el entorno como kernel de Jupyter (para que aparezca en VS Code y Jupyter):

```bash
python -m ipykernel install --user --name tema18 --display-name "Python (tema18)"
```

Para salir del entorno al terminar: `conda deactivate`.

### Opción B — venv (Python estándar)

**macOS / Linux**
```bash
cd "semana-09/Tema-18"
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
cd "semana-09\Tema-18"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## 3. Instalar dependencias

Con el entorno **activado**, actualiza `pip` e instala las librerías comunes:

```bash
pip install --upgrade pip
pip install jupyter ipykernel transformers Pillow requests
```

Luego instala **PyTorch** según tu hardware:

- **Con GPU NVIDIA (CUDA 12.1)**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- **Solo CPU** (o Apple Silicon)
  ```bash
  pip install torch torchvision
  ```

### 3.1 Tabla de paquetes

| Paquete | Uso |
| --- | --- |
| `transformers` | Provee las clases `CLIPModel` y `CLIPProcessor` y descarga el modelo preentrenado desde Hugging Face. |
| `torch`, `torchvision` | Backend de *deep learning* sobre el que corre CLIP (tensores, `softmax`, inferencia). |
| `Pillow` (`PIL`) | Abrir y decodificar la imagen de entrada. |
| `requests` | Descargar la imagen de ejemplo desde internet. |
| `jupyter`, `ipykernel` | Ejecutar el notebook. |

---

## 4. Imagen de entrada

Por defecto el notebook **descarga desde internet** una imagen de ejemplo (dos gatos) del dataset **COCO** (`http://images.cocodataset.org/val2017/000000039769.jpg`), la misma que usa la documentación oficial de CLIP. No necesitas ningún archivo local para probarlo.

Si prefieres usar **tu propia imagen**, coloca un archivo (`.jpg`, `.png`, etc.) en esta carpeta y, en el notebook, descomenta la línea `image = Image.open("imagen.jpg")` (ajustando el nombre) y comenta las líneas que descargan la imagen desde `url`.

---

## 5. Ejecutar el notebook

1. Abre [`Tema_18.ipynb`](Tema_18.ipynb) en VS Code o Jupyter.
2. Selecciona el kernel del entorno que creaste (**Python (tema18)** o `.venv`).
3. Ejecuta la celda de código. La **primera** ejecución descarga el modelo (~600 MB), por lo que puede tardar unos minutos.

---

## 6. Solución de problemas

| Problema | Causa / solución |
| --- | --- |
| Error de red al descargar la imagen de ejemplo | Revisa tu conexión, o usa una imagen local (ver [sección 4](#4-imagen-de-entrada)). |
| Descarga lenta o error de red al cargar el modelo | Es la descarga inicial desde Hugging Face. Reintenta; una vez descargado queda en caché local. |
| `ModuleNotFoundError: No module named 'transformers'` / `torch` | El entorno no está activado o faltan dependencias. Repite la [sección 3](#3-instalar-dependencias). |
| El kernel no aparece en VS Code | Registra el kernel con `python -m ipykernel install --user --name tema18` y recarga la ventana. |
