# -*- coding: utf-8 -*-
"""
Pipeline básico de NLP en español con spaCy, NLTK, Doc2Vec y Word2Vec.
Código extraído y unificado a partir de las imágenes compartidas por el usuario.
"""

# https://data.mendeley.com/datasets/j4pxzxpkc3/1

import pandas as pd
import spacy
from unidecode import unidecode
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
import nltk


# ================================
# 1. Carga de datos
# ================================
data = pd.read_csv("data/MexPol_Tweets.csv")
print(data.describe())


# ================================
# 2. Recursos de lenguaje
# ================================
# Descargar lista de stopwords de NLTK (solo la primera vez)
nltk.download("stopwords")

nlp = spacy.load("es_core_news_md")
stopwords_es = set(stopwords.words("spanish"))

neg_keep = {"no", "ni", "tampoco"}

# Quitamos las negaciones de la lista de stopwords para que no se eliminen
stopwords_es = stopwords_es - neg_keep


# ================================
# 3. Limpieza de texto
# ================================
def clean_text(texto: str, return_str: bool = False):
    """
    Limpia un texto en español (Latinoamérica) usando spaCy + NLTK stopwords:

    - Tokenización
    - Elimina espacios y signos de puntuación
    - Elimina stopwords (usando la lista de NLTK, conservando las negaciones)
    - Lematiza las palabras
    - Convierte a minúsculas y quita acentos

    Devuelve una lista de tokens limpios o una cadena si return_str=True.
    """
    doc = nlp(texto)

    tokens = [
        unidecode(t.lemma_.lower())
        for t in doc
        if not (t.is_space or t.is_punct) and (t.lemma_.lower() not in stopwords_es)
    ]

    return " ".join(tokens) if return_str else tokens


# ================================
# 4. Prueba rápida
# ================================
texto = (
    "El lenguaje me permite transformar pensamientos en significado; "
    "inspiración y aprendizaje. Me siento pleno y feliz."
)

print(clean_text(texto))          # lista de tokens
print(clean_text(texto, True))    # string limpio


# ================================
# 5. Tokenización del dataset
# ================================
data["tokenized_text"] = data["tweet"].apply(lambda x: clean_text(x))

tagged_docs = [
    TaggedDocument(words=toks, tags=[f"DOC_{i}"])
    for i, toks in enumerate(data["tokenized_text"])
]


# ================================
# 6. Entrenamiento Doc2Vec
# ================================
# Hiperparámetros razonables para empezar:
# dm=1 (PV-DM), vector_size=100, window=5, min_count=2
# epochs=40-80

model = Doc2Vec(
    vector_size=50,
    window=3,
    min_count=2,
    workers=4,
    dm=1,          # 1 = PV-DM, 0 = PV-DBOW
    hs=0,
    negative=5,
    alpha=0.025,
    min_alpha=0.0005,
    epochs=60
)

# Construir vocabulario y entrenar
model.build_vocab(tagged_docs)
model.train(
    tagged_docs,
    total_examples=model.corpus_count,
    epochs=model.epochs
)


# ================================
# 7. Inferencia y similitud
# ================================
texto_nuevo = "El ine es bueno"
toks_nuevos = clean_text(texto_nuevo)

# Inferir vector de documento para el nuevo texto
vec_inferido = model.infer_vector(toks_nuevos)

# Buscar los documentos más similares en el espacio Doc2Vec
similares = model.dv.most_similar([vec_inferido], topn=2)

print("Tokens limpios (nuevo texto):", toks_nuevos)
print("\nTop-2 documentos más similares:")

for tag, score in similares:
    idx = int(tag.split("_")[1])
    print(f"{tag}  score={score:.4f}  ->  '{data['tweet'][idx]}'")


# ================================
# 8. Entrenamiento Word2Vec
# ================================
# Opción principal: Skip-gram (sg=1) suele capturar mejor
# palabras raras/relaciones semánticas.

w2v = Word2Vec(
    sentences=data["tokenized_text"].to_list(),
    vector_size=70,   # 100-300 es común
    window=2,         # contexto
    min_count=2,      # ignora palabras raras (<2 ocurrencias)
    workers=4,
    sg=1,             # 1=Skip-gram, 0=CBOW
    negative=10,      # negative sampling
    epochs=50,
    seed=42
)


# ================================
# 9. Consultas de ejemplo
# ================================
def show_similares(model, palabra, topn=10):
    if palabra in model.wv.key_to_index.keys():
        print(f"\nPalabras más similares a '{palabra}':")
        for w, s in model.wv.most_similar(palabra, topn=topn):
            print(f"{w:15s} {s:.4f}")
    else:
        print(f"\n'{palabra}' no está en el vocabulario.")


# Similitud de palabras
show_similares(w2v, "ine", topn=3)
show_similares(w2v, "mexico", topn=3)


# ================================
# 10. Requisitos sugeridos
# ================================
# pip install pandas spacy gensim nltk unidecode
# python -m spacy download es_core_news_md
