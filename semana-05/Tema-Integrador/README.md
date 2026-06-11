# Tema Integrador — Visión Computacional

Notebook integrador que combina y profundiza los conceptos de **Tema 9** (fundamentos) y **Tema 10** (preprocesamiento y aumentación). Diseñado para estudiantes de **maestría** con:

- Fundamentos matemáticos (espacios de color, convolución, transformaciones afines).
- **Widgets interactivos** (`ipywidgets`) para explorar parámetros en tiempo real.
- Comparaciones visuales lado a lado.
- Métricas cuantitativas (PSNR para denoising).
- Pipeline profesional con **Albumentations**.
- Ejercicios propuestos al final.

## Cómo ejecutar

- **Local** (VS Code / Jupyter): instala `requirements.txt` del repo raíz.
- **Google Colab**: abre con el badge "Open In Colab" del notebook. La celda de Setup descarga la imagen e instala lo necesario.

## Contenido

1. La imagen digital: muestreo, cuantización y tipos de dato
2. Espacios de color (RGB, HSV, Lab, YCrCb) — interactivo
3. Histogramas, ecualización y CLAHE — interactivo
4. Filtrado, gradientes y bordes (Sobel, Laplaciano, Canny) — interactivo
5. Segmentación por color en HSV — interactivo
6. Preprocesamiento para modelos: resize, crop, letterbox, normalización ImageNet
7. Aumentación geométrica con transformaciones afines — interactivo
8. Aumentación fotométrica — interactivo
9. Modelos de ruido y filtros denoising con métricas PSNR
10. Pipeline de aumentación con Albumentations
11. Estadísticas de un mini-dataset
12. Ejercicios propuestos
