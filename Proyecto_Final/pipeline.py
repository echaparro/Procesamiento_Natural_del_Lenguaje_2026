import torch


def get_pipeline(modelo, question, context):
    if modelo is None:
        return "No hay modelo cargado"
    if not question or not context:
        return "Escribe una pregunta y asegurate de tener un archivo cargado"

    lineas = []
    lineas.append(f"**Pregunta:** {question}")
    lineas.append(f"**Contexto:** _{context[:200]}..._" if len(context) > 200 else f"**Contexto:** _{context}_")
    lineas.append("")
    lineas.append("---")
    lineas.append("")

    tokenizer = modelo.tokenizer
    model = modelo.model

    # --- Step 1: Tokenization ---
    lineas.append("### Paso 1: Tokenizacion")
    lineas.append("El tokenizer convierte la pregunta y el contexto en IDs numericos, "
                  "agregando tokens especiales `[CLS]` y `[SEP]`.")

    inputs = tokenizer(
        question, context,
        return_tensors="pt",
        truncation="only_second",
        max_length=512,
        padding=True,
    )

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    n = len(tokens)
    lineas.append(f"- **Tokens totales:** {n}")
    lineas.append(f"- **Input IDs:** `{inputs['input_ids'][0].tolist()}`")
    lineas.append(f"- **Tokens:** `{tokens}`")
    type_ids = inputs.get("token_type_ids")
    if type_ids is not None:
        sep_idx = (type_ids[0] == 1).nonzero(as_tuple=True)[0][0].item()
        lineas.append(f"- **Segmento A (pregunta):** tokens 0 a {sep_idx - 1}")
        lineas.append(f"- **Segmento B (contexto):** tokens {sep_idx} a {n - 1}")
    lineas.append("")
    lineas.append("---")
    lineas.append("")

    # --- Step 2: Model inference ---
    lineas.append("### Paso 2: Inferencia (forward pass)")
    lineas.append("Los tokens pasan por las capas del transformer. "
                  "La salida son dos vectores de logits: uno para la posicion de inicio "
                  "y otro para la posicion de fin de la respuesta.")

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    lineas.append(f"- **Start logits shape:** {list(start_logits.shape)} (1 logit por token)")
    lineas.append(f"- **End logits shape:** {list(end_logits.shape)}")
    lineas.append(f"- **Start logits (raw):** `{start_logits.tolist()}`")
    lineas.append(f"- **End logits (raw):** `{end_logits.tolist()}`")
    lineas.append("")
    lineas.append("---")
    lineas.append("")

    # --- Step 3: Softmax & Argmax ---
    lineas.append("### Paso 3: Softmax y seleccion de posiciones")
    lineas.append("Se aplica softmax para convertir logits en probabilidades, "
                  "luego se elige el token con mayor probabilidad para inicio y fin.")

    start_probs = torch.softmax(start_logits, dim=0)
    end_probs = torch.softmax(end_logits, dim=0)

    start_idx = torch.argmax(start_probs).item()
    end_idx = torch.argmax(end_probs).item()

    # Top 5 candidates
    topk = min(5, n)
    top_start = torch.topk(start_probs, topk)
    top_end = torch.topk(end_probs, topk)

    lineas.append(f"**Top-{topk} inicio:**")
    lineas.append(f"| Token | Probabilidad |")
    lineas.append(f"|---|---|")
    for score, idx in zip(top_start.values, top_start.indices):
        tok = tokens[idx]
        lineas.append(f"| `{tok}` (pos {idx}) | {score:.4f} |")

    lineas.append("")
    lineas.append(f"**Top-{topk} fin:**")
    lineas.append(f"| Token | Probabilidad |")
    lineas.append(f"|---|---|")
    for score, idx in zip(top_end.values, top_end.indices):
        tok = tokens[idx]
        lineas.append(f"| `{tok}` (pos {idx}) | {score:.4f} |")

    lineas.append("")
    lineas.append(f"- **Inicio elegido:** posicion {start_idx} (`{tokens[start_idx]}`)")
    lineas.append(f"- **Fin elegido:** posicion {end_idx} (`{tokens[end_idx]}`)")
    lineas.append("")
    lineas.append("---")
    lineas.append("")

    # --- Step 4: Decode ---
    lineas.append("### Paso 4: Decodificacion de la respuesta")
    lineas.append("Se extraen los tokens desde la posicion de inicio hasta la de fin, "
                  "se decodifican a texto y se calcula la confianza.")

    if end_idx < start_idx:
        lineas.append("⚠️ El token de fin esta antes del de inicio → respuesta vacia.")
        answer = ""
        confidence = 0.0
    else:
        answer_tokens = inputs["input_ids"][0][start_idx: end_idx + 1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
        confidence = (start_probs[start_idx] * end_probs[end_idx]).item()

        lineas.append(f"- **Tokens respuesta (IDs):** `{answer_tokens.tolist()}`")
        lineas.append(f"- **Tokens respuesta (texto):** `{tokenizer.convert_ids_to_tokens(answer_tokens)}`")

    lineas.append(f"- **Respuesta final:** _{answer}_" if answer else "- **Respuesta final:** *(vacia)*")
    lineas.append(f"- **Confianza:** {confidence:.4f} ({confidence * 100:.2f}%)")
    lineas.append("")
    lineas.append("---")
    lineas.append("### Resumen")
    lineas.append(f"> **Pregunta:** {question}")
    lineas.append(f"> **Respuesta:** {answer or '(no encontrada)'}")
    lineas.append(f"> **Confianza:** {confidence * 100:.2f}%")
    lineas.append(f"> **Modelo:** {modelo.model_choice.value}")

    return "\n".join(lineas)
