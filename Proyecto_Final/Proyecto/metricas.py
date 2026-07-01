import torch
from chat import chat, ModelName
import re
import string
from typing import Tuple, List, Set

TEST_SET = {
    "context": (
        "El español es un idioma romance originario de la península ibérica. "
        "Es hablado por aproximadamente 600 millones de personas en el mundo, "
        "lo que lo convierte en la segunda lengua materna más hablada después del mandarín."
    ),
    "qa": [
        ("¿Cuántas personas hablan español aproximadamente?", "600 millones"),
        ("¿De dónde es originario el español?", "península ibérica"),
        ("¿Qué tipo de idioma es el español?", "romance"),
        ("¿Cuál es la segunda lengua materna más hablada?", "español"),
    ],
}

_MODEL_CACHE = {}


def _get_model(name):
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = chat(name)
    return _MODEL_CACHE[name]


def _tokenize(text, normalize=False):
    if normalize:
        text = _normalize_answer(text)
    return text.lower().split()


def _normalize_answer(s: str) -> str:
    """
    Normaliza la respuesta para comparación justa
    """
    def remove_articles(text):
        # Eliminar artículos en español e inglés
        articles = re.compile(r'\b(a|an|the|el|la|los|las|un|una|unos|unas)\b', re.UNICODE)
        return re.sub(articles, ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    def remove_extra_spaces(text):
        return re.sub(r'\s+', ' ', text).strip()
    
    # Aplicar todas las normalizaciones
    s = lower(s)
    s = remove_articles(s)
    s = remove_punc(s)
    s = white_space_fix(s)
    s = remove_extra_spaces(s)
    
    return s

def _compute_metrics(pred: str, expected: str, normalize: bool = True) -> Tuple[float, float, float, float]:
    """
    Calcula métricas EM, F1, Precision y Recall entre predicción y esperado
    """
    # Manejar casos especiales
    if pred is None:
        pred = ""
    if expected is None:
        expected = ""
    
    # Tokenizar
    pt = _tokenize(pred, normalize)
    et = _tokenize(expected, normalize)
       
    # Calcular intersección
    common = set(pt) & set(et)
    
    # Calcular métricas
    precision = len(common) / len(pt) if len(pt) > 0 else 0.0
    recall = len(common) / len(et) if len(et) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Exact Match (normalizado)
    if normalize:
        pred_norm = _normalize_answer(pred)
        exp_norm = _normalize_answer(expected)
        em = 1.0 if pred_norm == exp_norm else 0.0
    else:
        em = 1.0 if pred.strip() == expected.strip() else 0.0
    
    return em, f1, precision, recall


def get_metricas(modelo):
    if modelo is None:
        return "No hay modelo cargado"

    config = modelo.model.config
    total = sum(p.numel() for p in modelo.model.parameters())
    treinable = sum(p.numel() for p in modelo.model.parameters() if p.requires_grad)

    def cfg(attr, fallback="—"):
        return getattr(config, attr, fallback)

    lineas = []
    lineas.append(f"**Modelo activo:** {modelo.model_choice.value}")
    lineas.append(f"**HF name:** `{modelo.hf_model_name}`")
    lineas.append(f"**Arquitectura:** {cfg('model_type')}")
    lineas.append(f"**Capas (hidden layers):** {cfg('num_hidden_layers')}")
    lineas.append(f"**Attention heads:** {cfg('num_attention_heads')}")
    lineas.append(f"**Hidden size:** {cfg('hidden_size')}")
    lineas.append(f"**Vocab size:** {cfg('vocab_size')}")
    lineas.append(f"**Max position embeddings:** {cfg('max_position_embeddings')}")
    lineas.append(f"**Parametros totales:** {total:,}")
    lineas.append(f"**Parametros entrenables:** {treinable:,}")
    lineas.append("")
    lineas.append(f"**Activacion:** {cfg('hidden_act')}")
    lineas.append(f"**Dropout:** {cfg('hidden_dropout_prob')}")
    lineas.append(f"**Attention dropout:** {cfg('attention_probs_dropout_prob')}")
    lineas.append("")
    lineas.append("---")

    
    lineas.append("### Evaluacion contra test set")
    lineas.append("")
    lineas.append(f"**Contexto usado:** _{TEST_SET['context']}_")
    lineas.append("")
    lineas.append("| # | Pregunta | Respuesta esperada |")
    lineas.append("|---|---|---|")
    for i, (q, a) in enumerate(TEST_SET["qa"], 1):
        lineas.append(f"| {i} | {q} | {a} |")
    lineas.append("")

    names = ["DistilBERT", "BERTIN"]
    agg = {n: {"EM": [], "F1": [], "Precision": [], "Recall": []} for n in names}

    for name in names: #Recorremos los dos modelos para preguntar y sacar las metricas
        m = _get_model(name)
        for question, expected in TEST_SET["qa"]:
            result = m.answer_question(question, TEST_SET["context"])
            pred = result.get("answer", "")
            em, f1, precision, recall = _compute_metrics(pred, expected)
            agg[name]["EM"].append(em)
            agg[name]["F1"].append(f1)
            agg[name]["Precision"].append(precision)
            agg[name]["Recall"].append(recall)

    lineas.append("| Metrica | Descripcion | DistilBERT | BERTIN |")
    lineas.append("|---|---|---|---|")
    rows = [
        ("EM", "Exact Match (coincidencia exacta)"),
        ("F1", "F1 Score (precision/recall de tokens)"),
        ("Precision", "Tokens correctos predichos"),
        ("Recall", "Tokens de respuesta encontrados"),
    ]
    for key, desc in rows:
        vals = []
        for name in names:
            avg = sum(agg[name][key]) / len(agg[name][key]) * 100
            vals.append(f"{avg:.1f}%")
        lineas.append(f"| **{key}** | {desc} | {vals[0]} | {vals[1]} |")

    return "\n".join(lineas)
