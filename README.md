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

1. Clona o abre el repositorio en VS Code.

2. Crea y activa un entorno virtual (recomendado):

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

4. Descarga el modelo de spaCy en español (usado en casi todos los notebooks):

   ```bash
   python -m spacy download es_core_news_md
   ```

5. Descarga los recursos de NLTK que se utilizan (stopwords):

   ```bash
   python -m nltk.downloader stopwords
   ```

   > Los notebooks también ejecutan `nltk.download("stopwords")` automáticamente la primera vez.

## Cómo ejecutar el código

### Opción A — VS Code (recomendada)

1. Abre la carpeta `02-Codigo` en VS Code.
2. Instala la extensión **Jupyter** si aún no la tienes.
3. Abre cualquier `.ipynb` (por ejemplo `semana-01/Tema-01/EjemploTema1.ipynb`).
4. En la esquina superior derecha del notebook, selecciona el kernel del entorno virtual `.venv` que creaste.
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

## Notas

- **Transformers / PyTorch**: el notebook `semana-02/Tema-03/Modelos generativos.ipynb` descarga modelos de Hugging Face (por ejemplo `bert-base-uncased`). La primera ejecución puede tardar varios minutos.
- **Rutas de datos**: cada notebook lee sus datos desde su carpeta `data/` relativa. Ejecuta los notebooks desde su propio directorio (VS Code lo hace por defecto) para que las rutas funcionen.
- **GPU (opcional)**: si tienes GPU NVIDIA, puedes reemplazar `torch` en `requirements.txt` por la build con CUDA siguiendo las instrucciones de https://pytorch.org/get-started/locally/.
