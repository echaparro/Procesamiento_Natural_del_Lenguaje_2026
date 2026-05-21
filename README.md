# Procesamiento Natural del Lenguaje (NLP)
### Maestría en Inteligencia Artificial · ALINNCO

---

## Descripción del Curso

Esta disciplina estudia el diseño de métodos y algoritmos que reciben como entrada y/o producen como salida datos en forma de lenguaje natural (e.g., texto, voz). El curso se centra en el procesamiento de texto aunque se
mencionan aplicaciones en procesamiento de voz.

---

`Objetivo General.` *Introducir al análisis y transformación del lenguaje mediante técnicas computacionales. Representar al texto y discurso que pueden ayudar a la predicción, la extracción y el razonamiento semántico sobre el lenguaje. Comprender el lenguaje natural para su aplicación en los hardwares y software.*

---
## Programa del Curso


`Introducción.` **Introducción a NLP**
##### Semana 1

##### Viernes · 4 hrs — Fundamentos
    1. ¿Qué es el NLP y sus aplicaciones?
    2. Conceptos clave: tokens, corpus, vocabulario
    3. Embeddings: intuición y uso
    4. Pipeline típico de un proyecto NLP
    5. Configuración del entorno (conda / venv)

##### Sábado · 4 hrs — Herramientas base
    1. Introducción a NLTK y spaCy
    2. Manejo de texto con `re` y `string`
    3. Exploración de un corpus real
    4. Ejercicio: limpieza y preprocesamiento básico
    5. Q&A y dudas de instalación

`Módulo 1.` **Preprocesamiento y Representación de Texto**

##### Semana 2 — Parte A: Procesamiento de texto

##### Viernes · 4 hrs — Transformaciones de texto
    1. Tokenización a nivel palabra y oración
    2. Stemming (PorterStemmer, SnowballStemmer)
    3. Lematización con spaCy y NLTK
    4.  Diferencias entre stemming y lematización
    5. Librerías: `NLTK`, `spaCy`

##### Sábado · 4 hrs — Limpieza y normalización
    1. Eliminación de stopwords
    2. Expresiones regulares para texto (`re`)
    3. Normalización: minúsculas, puntuación, acentos
    4. Construcción de un pipeline de limpieza
    5. Ejercicio: preprocesar corpus de noticias

##### Semana 3 — Parte B: Representación vectorial

##### Viernes · 4 hrs — Representaciones clásicas
    1. Bag of Words (BoW): construcción y limitaciones
    2. TF-IDF: cálculo e interpretación
    3. Visualización de matrices de términos
    4. Librerías: `scikit-learn`, `pandas`, `numpy`

##### Sábado · 4 hrs — Word Embeddings
    1. Word2Vec: CBOW vs Skip-gram
    2. GloVe y FastText: diferencias prácticas
    3. Sentence embeddings (`sentence-transformers`)
    4. Ejercicio: similitud semántica con Word2Vec
    5. Librerías: `gensim`, `sentence-transformers`

    
`Módulo 2.` **Tareas Clásicas de NLP**

##### Semana 4
##### Viernes · 4 hrs — Clasificación de texto
    1. Análisis de sentimientos (positivo / negativo / neutro)
    2. Clasificación multi-clase de documentos
    3. Modelos: Naive Bayes, SVM, Logistic Regression

##### Sábado · 4 hrs — Extracción de información
    1. Named Entity Recognition (NER)
    2. Part-of-Speech Tagging (POS)
    3. Extracción de keywords y relaciones

   
`Módulo 3.` **Modelos de Lenguaje y Etiquetado de Secuencia.**

##### Semana 5    

##### Viernes · 4 hrs — Arquitectura Transformer
    1. Mecanismo de atención (Self-Attention)
    2. Encoders y Decoders
    3. BERT y RoBERTa: modelos pre-entrenados
    4. Fine-tuning para tareas específicas
    5. Librerías: `transformers` (Hugging Face), `torch`, `datasets`

##### Sábado · 4 hrs — Proyecto Final
    1. Evaluación final 
    2. Presentación del proyecto
    3. Retroalimentación y cierre del curso       

**Recursos:**
- *Speech and Language Processing* — Jurafsky & Martin (Cap. 1-2)
- [NLTK Book](https://www.nltk.org/book/) — Capítulo 1
- Documentación oficial de spaCy: https://spacy.io
- *NLP with Python* — Bird, Klein & Loper (Cap. 3)
- Mikolov et al. (2013) — *Efficient Estimation of Word Representations in Vector Space*
- [Gensim Word2Vec Tutorial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html)
- [scikit-learn Text Classification Guide](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- *NLP with Python* — Bird, Klein & Loper (Cap. 5-7)
- Documentación de stanza: https://stanfordnlp.github.io/stanza
- Vaswani et al. (2017) — *Attention Is All You Need*
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) — Capítulos 1-4
- *Natural Language Processing with Transformers* — Tunstall, von Werra & Wolf
---

## ⚙️ Instalación del Entorno

### Prerrequisitos
- Anaconda o Miniconda: https://www.anaconda.com/download
- Git (opcional): https://git-scm.com/

### Pasos

```bash
# 1. Crea el entorno desde el archivo
conda env create -f environment.yml

# 2. Activa el entorno
conda activate nlp_curso_2026

# 3. Descarga modelos de spaCy
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm

# 4. Descarga recursos de NLTK
python -c "import nltk; [nltk.download(r) for r in ['punkt','punkt_tab','stopwords','wordnet','movie_reviews','averaged_perceptron_tagger','maxent_ne_chunker','words']]"

# 5. Registra el kernel en Jupyter
python -m ipykernel install --user --name nlp_curso_2026 --display-name "Python (NLP 2026)"

# 6. Inicia JupyterLab
jupyter notebook
```


