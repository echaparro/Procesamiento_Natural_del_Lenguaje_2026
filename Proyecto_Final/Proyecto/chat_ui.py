import gradio as gr
import os
import re
from chat import chat, ModelName
from metricas import get_metricas
from pipeline import get_pipeline

# =============================================================================
# CONFIGURACION
# =============================================================================
MAX_CHATS = 3                                                    # Maximo de chats simultaneos
FILES_DIR = os.path.join(os.path.dirname(__file__), "Files")     # Carpeta donde se almacenan los .txt


def _load_chats_from_files():
    """
    Escanea la carpeta Files/ y crea un chat por cada archivo .txt encontrado.
    Cada chat hereda el nombre del archivo (sin extension) y su contenido.
    Retorna: lista de dicts con id, name, history, file_name, file_content, model.
    """
    chats = []
    if not os.path.exists(FILES_DIR):
        return chats
    files = sorted(f for f in os.listdir(FILES_DIR) if f.endswith(".txt"))
    for i, fname in enumerate(files):
        name = os.path.splitext(fname)[0]
        file_path = os.path.join(FILES_DIR, fname)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            content = ""
        chats.append({"id": i + 1, "name": name, "history": [], "file_name": fname, "file_content": content})
    return chats


CHATS_INIT = _load_chats_from_files()
model_choices = [m.value for m in ModelName]
MODEL_LIST = []
currentModel = chat("DistilBERT")
MODEL_LIST.append(("DistilBERT", currentModel))

def _safe_filename(name):
    """Reemplaza caracteres no validos en nombres de archivo Windows por '_'."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def _next_id(chats):
    """Retorna el siguiente ID disponible para un nuevo chat."""
    return max([c["id"] for c in chats], default=0) + 1


def _radio(names, value):
    """
    Crea un componente Radio con la lista de nombres de chats.
    names: lista de strings (nombres de chats)
    value: nombre del chat activo
    """
    return gr.Radio(choices=names, value=value, label="Chats", interactive=True)


# =============================================================================
# Eventos de la interfaz
# =============================================================================

def create_chat(custom_name, chats, active_chat_name):
    """
    Crea un nuevo chat.
    - Si se supera MAX_CHATS, retorna sin crear.
    - Si custom_name esta vacio, asigna 'Chat N'.
    - Si el nombre ya existe, agrega sufijo numerico '(2)', '(3)', etc.
    Retorna: estado actualizado (chats, radio, activo, file_status, chatbot, create_name, file_input, current_chat, titulo).
    """
    if len(chats) >= MAX_CHATS:
        return chats, None, active_chat_name, "Maximo 3 chats", None, custom_name, None, None, None
    new_id = _next_id(chats)
    new_name = custom_name.strip() if custom_name and custom_name.strip() else f"Chat {new_id}"
    existing = [c["name"] for c in chats]
    if new_name in existing:
        suffix = 2
        while f"{new_name} ({suffix})" in existing:
            suffix += 1
        new_name = f"{new_name} ({suffix})"
    chat_obj = {"id": new_id, "name": new_name, "history": [], "file_name": None, "file_content": None}
    chats = chats + [chat_obj]
    names = [c["name"] for c in chats]
    return chats, _radio(names, new_name), new_name, "", [], "", None, chat_obj, f"Conversacion - {new_name}"


def delete_chat(chats, active_chat_name):
    """
    Elimina el chat activo y su archivo asociado en Files/.
    Si era el ultimo chat, deja el estado vacio.
    Retorna: estado actualizado (chats, radio, activo, file_status, chatbot, file_input, current_chat, titulo).
    """
    file_path = os.path.join(FILES_DIR, _safe_filename(active_chat_name) + ".txt")
    if os.path.exists(file_path):
        os.remove(file_path)
    new_chats = [c for c in chats if c["name"] != active_chat_name]
    if not new_chats:
        return [], _radio([], None), "", None, None, None, None, "Conversacion"
    new_active = new_chats[0]["name"]
    names = [c["name"] for c in new_chats]
    return new_chats, _radio(names, new_active), new_active, "", None, None, new_chats[0], f"Conversacion - {new_active}"


def select_chat(chat_name, chats):
    """
    Cambia al chat seleccionado y carga su historial, archivo y modelo.
    Retorna: (file_status, file_path, history, chat_name, current_chat, titulo).
    """
    if not chat_name or not chats:
        return None, None, [], chat_name, None, "Conversacion"
    for c in chats:
        if c["name"] == chat_name:
            status = c["file_name"] if c["file_name"] else None
            file_path = os.path.join(FILES_DIR, c["file_name"]) if c["file_name"] else None
            return status, file_path, c["history"], chat_name, c, f"Conversacion - {chat_name}"
    return None, None, [], chat_name, None, "Conversacion"


def upload_file(file, chats, active_chat_name, current_chat, model_choice):
    """
    Carga un archivo .txt al chat activo:
    - Copia el contenido a Files/{nombre_del_chat}.txt
    - Crea el modelo de QA con el contexto del archivo
    Retorna: estado actualizado (chats, file_status, current_chat, current_chat).
    """
    if file is None or not active_chat_name:
        return chats, None, None, None
    content = open(file, "r", encoding="utf-8").read()
    os.makedirs(FILES_DIR, exist_ok=True)
    dest_name = _safe_filename(active_chat_name) + ".txt"
    dest_path = os.path.join(FILES_DIR, dest_name)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(content)
    new_chats = []
    updated_chat = None
    for c in chats:
        if c["name"] == active_chat_name:
            c = dict(c)
            c["file_name"] = dest_name
            c["file_content"] = content
            updated_chat = c
        new_chats.append(c)
    return new_chats, dest_name, updated_chat, updated_chat


def send_message(message, history, chats, active_chat_name, current_chat):
    """
    Procesa el mensaje del usuario:
    - Verifica que haya un archivo cargado.
    - Si existe modelo, ejecuta answer_question().
    - Agrega pregunta y respuesta al historial.
    Retorna: (msg_limpiado, history, chats, file_status, current_chat).
    """
    global currentModel
    if not message or not active_chat_name:
        return "", history, chats, "", None
    if not current_chat or not current_chat["file_content"]:
        return "", history, chats, "Carga un archivo .txt primero", None
    if currentModel is None:
        return "", history, chats, "Error: modelo no cargado", None
    
    if current_chat:
        result = currentModel.answer_question(message, current_chat["file_content"])
    else:
        result = ""
    history = history + [{"role": "user", "content": message}]
    if result:
        history = history + [{"role": "assistant", "content": f"R: {result['answer']} (confianza: {result['score']:.2f}), {currentModel.get_model_name()}"}]
    new_chats = []
    updated_chat = None
    for c in chats:
        if c["name"] == active_chat_name:
            c = dict(c)
            c["history"] = history
            updated_chat = c
        new_chats.append(c)
    return "", history, new_chats, "", updated_chat


def clear_chat(chats, active_chat_name):
    """
    Limpia el historial del chat activo sin eliminar el archivo cargado.
    Retorna: (chats, file_status, chatbot_vacio, current_chat).
    """
    new_chats = []
    status = None
    updated_chat = None
    for c in chats:
        if c["name"] == active_chat_name:
            c = dict(c)
            c["history"] = []
            status = c["file_name"] if c["file_name"] else None
            updated_chat = c
        new_chats.append(c)
    return new_chats, status, [], updated_chat

def load_model_chat(new_model, chats, active_chat_name):
    global currentModel
    loaded = None
    for name, model in MODEL_LIST:
        if name == new_model:
            loaded = model
            break
    if loaded is None:
        loaded = chat(new_model)
        MODEL_LIST.append((new_model, loaded))
    currentModel = loaded

    print(f"Modelo actual {new_model}",currentModel)
    updated_chat = None
    for c in chats:
        if c["name"] == active_chat_name:
            updated_chat = c
            break

    title = f"Conversacion - {active_chat_name} - {new_model}" if active_chat_name else "Conversacion"
    return new_model, chats, updated_chat, title


def show_metricas():
    global currentModel
    return get_metricas(currentModel)


def run_pipeline(question, current_chat):
    global currentModel
    context = current_chat["file_content"] if current_chat and current_chat.get("file_content") else ""
    return get_pipeline(currentModel, question, context)


# =============================================================================
# INTERFAZ DE USUARIO (Gradio Blocks)
# =============================================================================

with gr.Blocks(title="Q&A Chatbot") as chatGui:
    # --- Estado inicial (cargado desde Files/) ---
    initial_names = [c["name"] for c in CHATS_INIT]
    initial_active = CHATS_INIT[0]["name"] if CHATS_INIT else None
    initial_status = CHATS_INIT[0]["file_name"] if CHATS_INIT and CHATS_INIT[0]["file_name"] else None
    initial_file_path = os.path.join(FILES_DIR, CHATS_INIT[0]["file_name"]) if CHATS_INIT and CHATS_INIT[0]["file_name"] else None
    initial_title = f"Conversacion - {initial_active}" if initial_active else "Conversacion"

    # Estados globales
    chats_state = gr.State(CHATS_INIT)                         # Lista completa de chats
    active_chat_state = gr.State(initial_active if initial_active else "")  # Nombre del chat activo
    current_chat = gr.State(CHATS_INIT[0] if CHATS_INIT else None)          # Objeto del chat activo
    model_choice = gr.State("DistilBERT")

    with gr.Row():
        # --- SIDEBAR (panel izquierdo) ---
        with gr.Column(scale=1, min_width=220):
            with gr.Row():
                create_name = gr.Textbox(label="Nuevo chat", placeholder="Nombre (opcional)...", scale=3, container=True)
                create_btn = gr.Button("+", variant="primary", scale=1, min_width=50)
            chat_list = _radio(initial_names, initial_active)  # Lista de chats
            gr.Markdown("---")
            delete_btn = gr.Button("Eliminar chat", variant="stop", size="sm")
            file_input = gr.File(label="Cargar archivo .txt", file_types=[".txt"], value=initial_file_path)
            file_status = gr.Markdown(initial_status if initial_status else "")
            model_dropdown = gr.Dropdown(choices=model_choices, value="DistilBERT", label="Modelo", interactive=True)

        # --- PANEL PRINCIPAL (tabs: Chat / Metricas) ---
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("Chat"):
                    gr.Markdown("## Chatbot de Preguntas y Respuestas (QA)")
                    chat_title = gr.Markdown(initial_title)
                    chatbot = gr.Chatbot(
                        label="Conversacion",
                        placeholder="Selecciona o crea un chat para empezar",
                        height=340,
                    )
                    with gr.Row():
                        msg = gr.Textbox(label="Tu pregunta", placeholder="Escribe tu mensaje aqui...", scale=4, container=True)
                        send = gr.Button("Enviar", variant="primary", scale=1, min_width=100)
                    with gr.Row():
                        clear = gr.Button("Limpiar conversacion", size="sm", variant="secondary")
                with gr.TabItem("Metricas"):
                    gr.Markdown("## Metricas del Modelo")
                    metrics_btn = gr.Button("Cargar Metricas", variant="primary")
                    metrics_md = gr.Markdown("Presiona 'Cargar Metricas' para ver las metricas")
                with gr.TabItem("Pipeline"):
                    gr.Markdown("## Pipeline del Modelo")
                    pipeline_q = gr.Textbox(label="Pregunta", placeholder="Escribe la pregunta para analizar el pipeline...")
                    pipeline_run = gr.Button("Ejecutar Pipeline", variant="primary")
                    pipeline_md = gr.Markdown("Escribe una pregunta y presiona 'Ejecutar Pipeline'")


    # Crear nuevo chat
    create_btn.click(
        create_chat,
        [create_name, chats_state, active_chat_state],
        [chats_state, chat_list, active_chat_state, file_status, chatbot, create_name, file_input, current_chat, chat_title],
    )

    # Seleccionar chat de la lista
    chat_list.input(
        select_chat, [chat_list, chats_state],
        [file_status, file_input, chatbot, active_chat_state, current_chat, chat_title],
    )

    # Eliminar chat activo
    delete_btn.click(
        delete_chat, [chats_state, active_chat_state],
        [chats_state, chat_list, active_chat_state, file_status, chatbot, file_input, current_chat, chat_title],
    )

    # Subir archivo .txt
    file_input.upload(
        upload_file, [file_input, chats_state, active_chat_state, current_chat, model_choice],
        [chats_state, file_status, current_chat, current_chat],
    )

    # Cambiar modelo seleccionado
    model_dropdown.input(
        load_model_chat, [model_dropdown, chats_state, active_chat_state],
        [model_choice, chats_state, current_chat, chat_title],
    )

    # Mostrar metricas del modelo actual (via tab Metricas)
    metrics_btn.click(show_metricas, None, [metrics_md])

    # Mostrar pipeline del modelo (via tab Pipeline)
    pipeline_run.click(run_pipeline, [pipeline_q, current_chat], [pipeline_md])

    # Enviar pregunta (clic en boton o tecla Enter)
    send.click(
        send_message, [msg, chatbot, chats_state, active_chat_state, current_chat],
        [msg, chatbot, chats_state, file_status, current_chat],
    )
    msg.submit(
        send_message, [msg, chatbot, chats_state, active_chat_state, current_chat],
        [msg, chatbot, chats_state, file_status, current_chat],
    )

    # Limpiar historial del chat activo
    clear.click(
        clear_chat, [chats_state, active_chat_state],
        [chats_state, file_status, chatbot, current_chat],
    )

if __name__ == "__main__":
    chatGui.launch(footer_links=[])
