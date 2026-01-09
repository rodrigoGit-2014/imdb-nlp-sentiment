# 🎬 IMDB NLP Sentiment Classification (Tarea NLP – 3 Modelos)

Este repositorio implementa y compara **tres enfoques** para **clasificación de sentimiento** (positivo/negativo) sobre reseñas IMDB, siguiendo un flujo NLP completo desde texto crudo hasta métricas de evaluación:

- **Modelo A (clásico):** TF-IDF + clasificador tradicional  
- **Modelo B (DL):** Embeddings estáticos + RNN  
- **Modelo C (Transformer):** BERT (fine-tuning)

> Nota importante: este README está construido **a partir de la estructura del proyecto** (carpetas/archivos). Los hiperparámetros y rutas exactas se controlan en `configs/*.yaml`.

---

## ✅ Requisitos

- Conda (recomendado) o Python compatible con tu entorno.
- Dependencias definidas en:
  - `environment.yml` (entorno)
  - `pyproject.toml` (paquete / tooling)

---

## ⚙️ Instalación

### 1) Crear y activar el entorno (Conda)

```bash
conda env create -f environment.yml
conda activate imdb-nlp
```

### 2) Instalar el proyecto en modo editable

Desde la raíz del repo:

```bash
pip install -e .
```

---

## 📁 Estructura del proyecto

```
├── configs
│   ├── base.yaml
│   ├── dataset.yaml
│   ├── model_a_tfidf.yaml
│   ├── model_b_static_emb.yaml
│   └── model_c_bert.yaml
├── data
├── environment.yml
├── notebooks
├── pyproject.toml
├── src
│   └── nlp_imdb
│       ├── cli
│       │   └── main.py
│       ├── data
│       │   ├── dataset_contract.md
│       │   ├── dataset_loader.py
│       │   └── splits.py
│       ├── features
│       │   ├── embeddings_static.py
│       │   └── tfidf.py
│       ├── models
│       │   ├── base.py
│       │   ├── model_a_tfidf.py
│       │   ├── model_b_rnn_static.py
│       │   └── model_c_bert.py
│       ├── preprocessing
│       │   ├── text_cleaning.py
│       │   └── tokenization.py
│       ├── training
│       │   ├── metrics.py
│       │   ├── result.py
│       │   ├── train_a.py
│       │   ├── train_b.py
│       │   ├── train_c.py
│       │   └── trainer.py
│       └── utils
│           ├── logging.py
│           ├── paths.py
│           └── seed.py
└── tests
    ├── conftest.py
    └── test_smoke.py
```

---

## 🧩 ¿Qué hace cada módulo?

### CLI (punto de entrada)
- `src/nlp_imdb/cli/main.py`  
  Ejecuta etapas (`--stage`) y carga configuración (`--config`). Es el comando que usarás para entrenar cada modelo.

### Datos
- `src/nlp_imdb/data/dataset_contract.md`  
  Contrato/expectativas del dataset (formato esperado).
- `src/nlp_imdb/data/dataset_loader.py`  
  Carga el dataset y lo prepara para entrenamiento/evaluación.
- `src/nlp_imdb/data/splits.py`  
  Lógica para particionar en train/val/test de forma reproducible.

### Preprocesamiento
- `src/nlp_imdb/preprocessing/text_cleaning.py`  
  Limpieza del texto (ruido básico: símbolos, normalización, etc.).
- `src/nlp_imdb/preprocessing/tokenization.py`  
  Tokenización y utilidades relacionadas (especialmente útil para el Modelo B y/o C).

### Features
- `src/nlp_imdb/features/tfidf.py`  
  Construcción de features TF-IDF (Modelo A).
- `src/nlp_imdb/features/embeddings_static.py`  
  Embeddings estáticos / matriz de embeddings (Modelo B).

### Modelos
- `src/nlp_imdb/models/base.py`  
  Interfaz/clase base común (si aplica).
- `src/nlp_imdb/models/model_a_tfidf.py`  
  Definición del modelo A (pipeline clásico).
- `src/nlp_imdb/models/model_b_rnn_static.py`  
  Definición del modelo B (RNN + embeddings).
- `src/nlp_imdb/models/model_c_bert.py`  
  Definición del modelo C (BERT / Transformer).

### Entrenamiento y evaluación
- `src/nlp_imdb/training/trainer.py`  
  Orquestador del entrenamiento (fit/evaluate, logging, guardado, etc.).
- `src/nlp_imdb/training/train_a.py` / `train_b.py` / `train_c.py`  
  Pipelines específicos por modelo.
- `src/nlp_imdb/training/metrics.py`  
  Métricas (accuracy, precision, recall, f1, etc.).
- `src/nlp_imdb/training/result.py`  
  Estructuras de salida/resultados (formato final de reporting).

### Utilidades
- `src/nlp_imdb/utils/logging.py`  
  Configuración de logs.
- `src/nlp_imdb/utils/paths.py`  
  Manejo de rutas (data, outputs, etc.).
- `src/nlp_imdb/utils/seed.py`  
  Semillas para reproducibilidad.

### Tests
- `tests/test_smoke.py`  
  Prueba mínima para verificar que el pipeline “enciende” correctamente.
- `tests/conftest.py`  
  Fixtures de pytest.

---

## 🚀 Cómo ejecutar cada modelo

El comando general es:

```bash
python -m nlp_imdb.cli.main --config <ruta_config.yaml> --stage <stage>
```

### ▶ Modelo A (TF-IDF)

```bash
python -m nlp_imdb.cli.main --config configs/model_a_tfidf.yaml --stage train_a
```

### ▶ Modelo B (Embeddings + RNN)

```bash
python -m nlp_imdb.cli.main --config configs/model_b_static_emb.yaml --stage train_b
```

### ▶ Modelo C (BERT)

```bash
python -m nlp_imdb.cli.main --config configs/model_c_bert.yaml --stage train_c
```

---

## 🧠 Flujo NLP que demuestras en la tarea

### Modelo A (Clásico)
1. Texto crudo → limpieza (`text_cleaning.py`)
2. Vectorización TF-IDF (`tfidf.py`)
3. Entrenamiento clasificador (`model_a_tfidf.py` + `train_a.py`)
4. Métricas (`metrics.py`)

### Modelo B (Deep Learning)
1. Texto crudo → limpieza
2. Tokenización + vocabulario (`tokenization.py`)
3. Embeddings estáticos (`embeddings_static.py`)
4. RNN (p.ej. LSTM) (`model_b_rnn_static.py` + `train_b.py`)
5. Métricas

### Modelo C (Transformer)
1. Texto crudo → tokenización tipo BERT (en el pipeline del modelo)
2. Fine-tuning de BERT (`model_c_bert.py` + `train_c.py`)
3. Métricas

---

## 🧪 Ejecutar pruebas

```bash
pytest -q
```

---

## 🧹 Sobre `__pycache__` y `.pyc`

Al ejecutar Python, se generan carpetas `__pycache__` y archivos `.pyc` (bytecode) automáticamente para acelerar imports.  
No deben subirse al repo; típicamente se agregan al `.gitignore`:

```gitignore
__pycache__/
*.pyc
```

---

## 👤 Autor

Rodrigo Cáceres – Magíster en Data Science

---

## 📄 Licencia

Uso académico (tarea).
