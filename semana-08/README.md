# Semana 08 — Procesamiento inteligente de documentos y despliegue de modelos

Guía para ejecutar **localmente** los notebooks de esta semana:

- **`Tema-15/Tema_15.ipynb`** — Extracción de datos de facturas con **OCR** (`docTR`) y un **LLM** local servido con **Ollama**.
- **`Tema-16/Tema_16.ipynb`** — Despliegue de un modelo como **API REST** con **Flask** + **TensorFlow Serving**.

> A diferencia de otras semanas, estos notebooks **no** traen el badge *Open In Colab*, ya que dependen de **servicios locales** (Ollama y TensorFlow Serving) que no están disponibles en Google Colab.

---

## 1. Requisitos

- **Python 3.10 – 3.12**
- ~6 GB de espacio libre (modelos de docTR/PyTorch y, para Tema 15, el modelo del LLM en Ollama).
- (Opcional pero recomendado) **GPU NVIDIA** con **CUDA 12.1** para acelerar el OCR de Tema 15. Sin GPU los notebooks funcionan igual, pero más lento.
- Servicios adicionales según el tema:
  - **Tema 15:** [Ollama](https://ollama.com) instalado y en ejecución.
  - **Tema 16:** [Docker](https://docs.docker.com/get-docker/) para levantar **TensorFlow Serving** (opcional, solo si quieres probar la inferencia real).

---

## 2. Crear un entorno

Puedes usar **Conda** (recomendada) o **venv**. Elige una sola opción.

### Opción A — Conda / Miniconda

```bash
cd "semana-08"
conda create -n semana08 python=3.11 -y
conda activate semana08
```

Para registrar el entorno como kernel de Jupyter (para que aparezca en VS Code y Jupyter):

```bash
python -m ipykernel install --user --name semana08 --display-name "Python (semana08)"
```

Para salir del entorno al terminar: `conda deactivate`.

### Opción B — venv (Python estándar)

**macOS / Linux**
```bash
cd "semana-08"
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
cd "semana-08"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## 3. Instalar dependencias

Con el entorno **activado**, actualiza `pip` e instala primero las librerías comunes:

```bash
pip install --upgrade pip
pip install jupyter ipykernel numpy pandas Pillow matplotlib mplcursors scikit-learn requests
```

### 3.1 Tema 15 — OCR (`docTR`) + PyTorch

```bash
pip install python-doctr datasets
```

Instala **PyTorch** según tu hardware:

- **Con GPU NVIDIA (CUDA 12.1)** — igual que en el notebook:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- **Solo CPU** (o Apple Silicon):
  ```bash
  pip install torch torchvision
  ```

### 3.2 Tema 16 — API REST con Flask

```bash
pip install flask opencv-python
```

### 3.3 Tabla de paquetes

| Paquete | Notebook | Uso |
| --- | --- | --- |
| `python-doctr` | Tema 15 | Motor de **OCR** que reconoce el texto y las coordenadas de la factura. |
| `torch`, `torchvision` | Tema 15 | Backend de *deep learning* que usa docTR. |
| `datasets` | Tema 15 | Descarga el dataset `mauroibz/facturas-argentinas` desde Hugging Face. |
| `pandas`, `Pillow`, `matplotlib`, `mplcursors` | Tema 15 | Manipulación de datos y visualización de las imágenes. |
| `requests` | Tema 15 y 16 | Peticiones HTTP a Ollama (Tema 15) y a TensorFlow Serving (Tema 16). |
| `flask` | Tema 16 | Framework para exponer el modelo como **API REST**. |
| `opencv-python` | Tema 16 | Decodificación y preprocesamiento de las imágenes recibidas. |
| `numpy` | Tema 16 | Manejo de arreglos de imágenes. |
| `jupyter`, `ipykernel` | Ambos | Ejecutar los notebooks. |

---

## 4. Servicios locales adicionales

### 4.1 Ollama (necesario para **Tema 15**)

El notebook envía el texto de la factura a un **LLM** servido por Ollama en `http://localhost:11434`.

1. Instala Ollama desde <https://ollama.com/download> (macOS, Linux y Windows).
2. Inicia el servicio (en macOS/Windows basta con abrir la app; en Linux):
   ```bash
   ollama serve
   ```
3. Descarga el modelo que se usa en la **última celda** del notebook:
   ```bash
   ollama pull qwen3.5:latest
   ```
   > 💡 Si ese modelo no está disponible en tu versión de Ollama, usa cualquier modelo instalado (por ejemplo `ollama pull qwen2.5` o `ollama pull llama3.2`) y actualiza el campo `"model"` en la celda **"Conexión con la API local de Ollama"**.
4. Verifica que responde:
   ```bash
   ollama list
   ```

### 4.2 TensorFlow Serving (opcional, para **Tema 16**)

El notebook de Tema 16 es un **cliente**: la app de Flask reenvía la imagen a un servidor de **TensorFlow Serving** en `http://localhost:8501`. Para probar la inferencia real necesitas ese servicio con un modelo llamado `modelo`.

Si solo quieres revisar el código del notebook, puedes ejecutarlo **sin** este servicio: las peticiones a `/predict` fallarán con un error **`502`** controlado (es lo esperado).

#### Paso 1 — Descargar un modelo de ejemplo (MobileNetV2)

TensorFlow Serving necesita un **SavedModel** en la estructura `modelo/1/saved_model.pb`. Descarga uno de ejemplo (MobileNetV2 224×224, compatible con el preprocesamiento del notebook):

**macOS / Linux**
```bash
cd "semana-08/Tema-16"
mkdir -p tf_serving/modelo/1
curl -sL "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5?tf-hub-format=compressed" -o mobilenet.tar.gz
tar -xzf mobilenet.tar.gz -C tf_serving/modelo/1
rm mobilenet.tar.gz
```

**Windows (PowerShell)**
```powershell
cd "semana-08\Tema-16"
New-Item -ItemType Directory -Force -Path tf_serving\modelo\1 | Out-Null
Invoke-WebRequest -Uri "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5?tf-hub-format=compressed" -OutFile mobilenet.tar.gz
tar -xzf mobilenet.tar.gz -C tf_serving\modelo\1
Remove-Item mobilenet.tar.gz
```

> `tar` viene incluido en Windows 10/11. La carpeta `tf_serving/` (~15 MB) es un artefacto descargable; puedes **no** subirla al repositorio (agrégala a `.gitignore`) y volver a generarla con estos comandos.

#### Paso 2 — Levantar TensorFlow Serving con Docker

Instala **Docker** ([Docker Desktop](https://www.docker.com/products/docker-desktop/) en Windows/macOS, o `docker` en Linux) y asegúrate de que el daemon esté corriendo (`docker ps` no debe dar error). Luego, **desde la carpeta `semana-08/Tema-16`**:

**Linux / Windows / macOS con Intel** — imagen oficial (amd64):

*Linux / macOS (bash)*
```bash
docker run --rm --name tfserving -p 8501:8501 \
  --mount type=bind,source="$PWD/tf_serving/modelo",target=/models/modelo \
  -e MODEL_NAME=modelo \
  tensorflow/serving
```

*Windows (PowerShell)*
```powershell
docker run --rm --name tfserving -p 8501:8501 `
  --mount type=bind,source="${PWD}\tf_serving\modelo",target=/models/modelo `
  -e MODEL_NAME=modelo `
  tensorflow/serving
```

**macOS con Apple Silicon (M1/M2/M3)** — la imagen oficial es solo amd64. Usa **Docker Desktop para Apple Silicon** y una imagen **arm64 nativa** (más rápida):

```bash
docker run --rm --name tfserving -p 8501:8501 \
  --mount type=bind,source="$PWD/tf_serving/modelo",target=/models/modelo \
  -e MODEL_NAME=modelo \
  emacski/tensorflow-serving:latest-linux_arm64
```

> Alternativa en Apple Silicon: la imagen oficial emulada con `docker run --platform linux/amd64 ... tensorflow/serving` (funciona pero es más lenta). Ojo: el Docker Desktop **de Intel no arranca** en Macs con Apple Silicon; instala la build "Mac with Apple chip".

Cuando el contenedor esté listo verás en su log `Successfully loaded servable {name: modelo version: 1}`.

#### Iniciar y detener el contenedor

Los comandos de arriba corren en **primer plano** (ocupan la terminal). Para detenerlos, presiona **`Ctrl + C`** en esa terminal.

Si prefieres dejarlo **en segundo plano** (para seguir usando la terminal), agrega `-d` y un nombre con `--name`. Ejemplo en macOS Apple Silicon (ajusta la imagen según tu sistema):

```bash
# Iniciar en segundo plano
docker run -d --rm --name tfserving -p 8501:8501 \
  --mount type=bind,source="$PWD/tf_serving/modelo",target=/models/modelo \
  -e MODEL_NAME=modelo \
  emacski/tensorflow-serving:latest-linux_arm64
```

Gestión del contenedor:

```bash
docker ps                 # ver contenedores en ejecución (y sus puertos)
docker logs -f tfserving  # ver los logs en vivo (Ctrl+C solo cierra el log, no el contenedor)
docker stop tfserving     # detener el contenedor
```

> Como los comandos usan `--rm`, al detenerse el contenedor **se elimina** (no queda basura). Para volver a levantarlo, ejecuta de nuevo el mismo `docker run`. El **modelo** en `tf_serving/modelo/` no se borra: es un archivo local que se monta en el contenedor.

#### Alternativa sin Docker (cualquier sistema)

Si no puedes usar Docker, hay un pequeño servidor que **emula** la API REST de TensorFlow Serving usando TensorFlow directamente. Requiere `pip install tensorflow` y el modelo del Paso 1:

```bash
python "semana-08/Tema-16/tf_serving_local.py"
```

Sirve el mismo endpoint en `http://localhost:8501`, así que el notebook funciona igual (la inferencia la ejecuta TensorFlow real).

#### Paso 3 — Probar

Con TensorFlow Serving arriba (Docker o el servidor local) **y** la celda de Flask del notebook en ejecución (puerto `5001`), envía una imagen desde otra terminal:

```bash
curl -X POST -F "file=@/ruta/a/tu/imagen.jpg" http://127.0.0.1:5001/predict
```

Deberías recibir un **`200`** con las `predictions` (las 1001 clases de ImageNet de MobileNet).

---

## 5. Ejecutar los notebooks

### VS Code (recomendado)

1. Abre la carpeta `02-Codigo` en VS Code.
2. Instala la extensión **Jupyter** si no la tienes.
3. Abre `semana-08/Tema-15/Tema_15.ipynb` o `semana-08/Tema-16/Tema_16.ipynb`.
4. En la esquina superior derecha selecciona el kernel:
   - **Conda:** `Python (semana08)`.
   - **venv:** el intérprete dentro de `.venv`.
5. Ejecuta las celdas con `Shift + Enter`.

### Jupyter en el navegador

```bash
jupyter notebook
```

Luego abre el notebook que quieras desde el navegador.

---

## 6. Notas

- **Tema 15:** la primera ejecución descarga el dataset y los pesos de docTR, por lo que puede tardar. El notebook desactiva la verificación SSL para evitar el error `SSLCertVerificationError` al descargar los modelos. Asegúrate de que **Ollama** esté en ejecución **antes** de correr la última celda.
- **Tema 16:** la celda con `app.run(...)` **bloquea** el kernel mientras el servidor de Flask está activo. Ejecútala en una terminal aparte (o detén la celda para continuar). El endpoint `/predict` solo devolverá predicciones si **TensorFlow Serving** está corriendo; de lo contrario responde con un error `502` controlado.
