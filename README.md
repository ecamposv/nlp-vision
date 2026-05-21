# Procesamiento de Lenguaje Natural y Visión Computacional

Código de las clases del Tetramestre 02 (2026) del curso **Procesamiento de Lenguaje Natural y Visión Computacional** de TecMilenio.

## Contenido

```
semana-01/
  Tema-01/   Introducción a spaCy y embeddings (EjemploTema1.ipynb)
  Tema-02/   Pipeline NLP con Word2Vec y Doc2Vec
             - EjemploTema2.ipynb
             - Tema_2_codigo.ipynb
             - nlp_doc2vec_word2vec_pipeline.py
semana-02/
  Tema-03/   Clasificación de texto y modelos generativos
             - Tema3.ipynb              (TF-IDF + SVM / LogReg con spaCy)
             - Modelos generativos.ipynb (Hugging Face transformers / BERT)
  Tema-04/   Generación de texto (Generacion de texto.ipynb)
```

Cada tema incluye su propio directorio `data/` con los datasets que usan los notebooks (`MexPol_Tweets.csv`, `punta_cana.csv`).

## Requisitos

- Python 3.10 o superior
- `pip` actualizado
- ~3 GB libres en disco (los modelos de Hugging Face y spaCy ocupan espacio)

## Instalación

Elige **una** de las dos opciones para gestionar el entorno: Anaconda (recomendada si ya lo tienes instalado) o `venv` de Python estándar.

### Opción 1 — Anaconda / Miniconda

1. Instala [Anaconda](https://www.anaconda.com/download) o [Miniconda](https://docs.conda.io/en/latest/miniconda.html) si aún no lo tienes.

2. Clona o abre el repositorio en VS Code y abre una terminal en la carpeta `02-Codigo`.

3. Crea el entorno conda con Python 3.10 y actívalo:

   ```bash
   conda create -n nlp-vision python=3.10 -y
   conda activate nlp-vision
   ```

4. Instala las dependencias con `pip` dentro del entorno conda:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   > Alternativa: si prefieres instalar todo desde el canal `conda-forge`, puedes usar
   > `conda install -c conda-forge pandas numpy matplotlib scikit-learn spacy nltk gensim unidecode jupyter ipykernel transformers pytorch`.
   > Sin embargo, `pip install -r requirements.txt` es más sencillo y reproduce exactamente las versiones del proyecto.

5. Registra el entorno como kernel de Jupyter (para que aparezca en VS Code y en Jupyter):

   ```bash
   python -m ipykernel install --user --name nlp-vision --display-name "Python (nlp-vision)"
   ```

6. Descarga el modelo de spaCy en español y los recursos de NLTK (ver sección [Recursos adicionales](#recursos-adicionales)).

Para salir del entorno cuando termines: `conda deactivate`.

### Opción 2 — venv (Python estándar)

1. Clona o abre el repositorio en VS Code.

2. Crea y activa un entorno virtual:

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

3. Actualiza `pip` e instala las dependencias:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Descarga el modelo de spaCy y los recursos de NLTK (ver sección siguiente).

### Recursos adicionales

Con el entorno (conda o venv) ya activado, ejecuta:

```bash
python -m spacy download es_core_news_md
python -m nltk.downloader stopwords
```

> Los notebooks también ejecutan `nltk.download("stopwords")` automáticamente la primera vez.

## Cómo ejecutar el código

### Opción A — VS Code (recomendada)

1. Abre la carpeta `02-Codigo` en VS Code.
2. Instala la extensión **Jupyter** si aún no la tienes.
3. Abre cualquier `.ipynb` (por ejemplo `semana-01/Tema-01/EjemploTema1.ipynb`).
4. En la esquina superior derecha del notebook, selecciona el kernel que corresponda:
   - **Anaconda**: `Python (nlp-vision)` (el que registraste con `ipykernel install`).
   - **venv**: el intérprete dentro de `.venv`.
5. Ejecuta las celdas con `Shift + Enter`.

### Opción B — Jupyter en el navegador

```bash
jupyter notebook
```

Navega al notebook deseado dentro de `semana-01/` o `semana-02/`.

### Opción C — Scripts de Python

El archivo `semana-01/Tema-02/nlp_doc2vec_word2vec_pipeline.py` se ejecuta directamente:

```bash
cd semana-01/Tema-02
python nlp_doc2vec_word2vec_pipeline.py
```

### Opción D — Google Colab

Cada notebook incluye al inicio un badge **Open in Colab** y una celda **Setup para Google Colab**. Para usarlo:

1. Abre el notebook en GitHub y haz clic en el badge *Open in Colab* (o ve a https://colab.research.google.com → *GitHub* → pega la URL del notebook).
2. Ejecuta la primera celda de código (*Setup para Google Colab*). Detecta el entorno automáticamente, instala las dependencias faltantes, descarga el modelo de spaCy y baja los archivos de `data/` desde el repositorio.
3. Ejecuta el resto del notebook con normalidad.

> En entornos locales (VS Code / Jupyter) esa misma celda no hace nada y se puede saltar.

## Notas

- **Transformers / PyTorch**: el notebook `semana-02/Tema-03/Modelos generativos.ipynb` descarga modelos de Hugging Face (por ejemplo `bert-base-uncased`). La primera ejecución puede tardar varios minutos.
- **Rutas de datos**: cada notebook lee sus datos desde su carpeta `data/` relativa. Ejecuta los notebooks desde su propio directorio (VS Code lo hace por defecto) para que las rutas funcionen.
- **GPU (opcional)**: si tienes GPU NVIDIA, puedes reemplazar `torch` en `requirements.txt` por la build con CUDA siguiendo las instrucciones de https://pytorch.org/get-started/locally/.
