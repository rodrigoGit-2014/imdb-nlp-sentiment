# 🎬 IMDB NLP Sentiment Classification

Proyecto académico para **clasificación de sentimiento** en reseñas de películas IMDB utilizando **tres modelos NLP** con distintos enfoques:

1. **Modelo A:** TF-IDF + clasificador tradicional  
2. **Modelo B:** Embeddings estáticos + RNN  
3. **Modelo C:** Transformer preentrenado (BERT)

El objetivo es **comparar enfoques clásicos vs deep learning vs transformers** y demostrar el flujo completo de un sistema NLP desde texto crudo hasta predicción final.

---

## 📁 Estructura del proyecto

.
├── configs  
│   ├── base.yaml  
│   ├── dataset.yaml  
│   ├── model_a_tfidf.yaml  
│   ├── model_b_static_emb.yaml  
│   └── model_c_bert.yaml  
│  
├── data  
│  
├── environment.yml  
├── pyproject.toml  
│  
├── src  
│   └── nlp_imdb  
│       ├── cli  
│       │   └── main.py  
│       │  
│       ├── data  
│       │   ├── dataset_loader.py  
│       │   ├── splits.py  
│       │   └── dataset_contract.md  
│       │  
│       ├── preprocessing  
│       │   ├── text_cleaning.py  
│       │   └── tokenization.py  
│       │  
│       ├── features  
│       │   ├── tfidf.py  
│       │   └── embeddings_static.py  
│       │  
│       ├── models  
│       │   ├── base.py  
│       │   ├── model_a_tfidf.py  
│       │   ├── model_b_rnn_static.py  
│       │   └── model_c_bert.py  
│       │  
│       ├── training  
│       │   ├── trainer.py  
│       │   ├── train_a.py  
│       │   ├── train_b.py  
│       │   ├── train_c.py  
│       │   ├── metrics.py  
│       │   └── result.py  
│       │  
│       └── utils  
│           ├── logging.py  
│           ├── paths.py  
│           └── seed.py  
│  
├── notebooks  
└── tests  
    ├── conftest.py  
    └── test_smoke.py  

---

## 🧠 Descripción de los modelos

### 🔹 Modelo A – TF-IDF

Enfoque clásico:
- Limpieza de texto  
- Vectorización TF-IDF  
- Clasificador tradicional (Logistic Regression / SVM)  

Representa documentos como vectores numéricos ponderados por frecuencia.

---

### 🔹 Modelo B – Embeddings estáticos + RNN

Deep Learning:
- Tokenización  
- Construcción de vocabulario  
- Capa Embedding  
- Red neuronal recurrente (RNN / LSTM)  

Aprende representación semántica de palabras.

---

### 🔹 Modelo C – BERT (Transformer)

Modelo preentrenado:
- Tokenización propia de BERT  
- Fine-tuning  
- Clasificación directa  

Usa atención y contexto bidireccional.

---

## ⚙️ Instalación

### 1️⃣ Crear entorno

```bash
conda env create -f environment.yml
conda activate imdb-nlp
