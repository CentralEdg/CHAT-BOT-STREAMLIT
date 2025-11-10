
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.prompts import ChatPromptTemplate
import matplotlib.pyplot as plt
import unicodedata, re


st.set_page_config(page_title="Chatbot Analítico - Paciente Digital 2025", layout="wide")

# ------------------
# Utilidades
# ------------------

@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer, **kwargs):
    try:
        df = pd.read_csv(path_or_buffer, **kwargs)
        return df
    except Exception as e:
        st.error(f"Error cargando {path_or_buffer}: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def try_load_local_or_upload(filename: str, uploaded_file):
    if uploaded_file is not None:
        return load_csv(uploaded_file)
    # fallback local
    if os.path.exists(filename):
        return load_csv(filename)
    return pd.DataFrame()



def limpiar_pregunta(texto):
    # Elimina números iniciales tipo "33.", "34 -", "12–"
    return re.sub(r'^\s*\d+\s*[\.\-\–]\s*', '', texto).strip()


def canon(text: str) -> str:
    """
    Normaliza texto para comparar sin tildes, sin dobles espacios y sin diferencias de mayúsculas.
    """
    s = unicodedata.normalize("NFD", str(text).strip().lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")  # quita tildes
    s = re.sub(r"\s+", " ", s)  # colapsa espacios
    return s

def build_unified_df(resultados, preguntas, demograficos, padecimientos):
    df = resultados.copy()
    if not preguntas.empty:
        df = df.merge(preguntas, on="NO_Pregunta", how="left")
    if not demograficos.empty:
        df = df.merge(demograficos, on="respondent_id", how="left")
    if not padecimientos.empty:
        df = df.merge(padecimientos, on="respondent_id", how="left")
    return df

def get_sections_sorted(preguntas_df):
    if "Seccion" in preguntas_df.columns and not preguntas_df.empty:
        secciones = sorted(pd.Series(preguntas_df["Seccion"].dropna().unique()).astype(str).tolist())
    else:
        secciones = []
    return secciones

def format_section_menu(secciones):
    if not secciones:
        # Solo la opción final si no hay secciones
        return ["Hacer una pregunta concreta"]
    opciones = [s for s in secciones]
    opciones.append("Hacer una pregunta concreta")
    return opciones

def azure_llm_from_env(temperature: float = 0.2):
    load_dotenv()
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    if not all([api_key, endpoint, api_version, deployment]):
        st.warning("Faltan variables de entorno de Azure OpenAI. Verifica tu archivo .env.")
    llm = AzureChatOpenAI(
        azure_deployment=deployment,
        openai_api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
        temperature=temperature,
    )
    return llm

def make_agent(df, llm):
    # Crea un agente de análisis sobre DataFrame unificado
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True  # 👈 necesario desde LangChain 0.2+
    )
    return agent

def section_overview(df_section: pd.DataFrame, section_name: str):
    st.subheader(f"Visión general – {section_name}")

    # Total de respondientes en la sección
    if "respondent_id" in df_section.columns:
        total_resp = df_section["respondent_id"].nunique()
        st.markdown(f"**Respondientes únicos en la sección:** {total_resp}")
    else:
        st.markdown("**Respondientes únicos en la sección:** N/D")

    # Top opciones por pregunta (si existen columnas)
    cols_ok = all(c in df_section.columns for c in ["NO_Pregunta", "Pregunta", "Opcion"])
    if cols_ok and not df_section.empty:
        st.markdown("**Top respuestas por pregunta (conteos):**")
        top_table = (
            df_section.groupby(["NO_Pregunta", "Pregunta", "Opcion"], dropna=False)
            .size()
            .reset_index(name="conteo")
            .sort_values(["NO_Pregunta", "conteo"], ascending=[True, False])
        )
        # Mostrar las 5 más frecuentes por pregunta
        top_5 = top_table.groupby("NO_Pregunta").head(5)
        st.dataframe(top_5, use_container_width=True)

        # Gráfico simple de la pregunta más respondida por volumen
        agg_p = (
            df_section.groupby(["NO_Pregunta", "Pregunta"], dropna=False)
            .size()
            .reset_index(name="respuestas")
            .sort_values("respuestas", ascending=False)
        )
        if not agg_p.empty:
            top_q = agg_p.iloc[0]
            st.markdown(f"**Pregunta con más respuestas:** {int(top_q['NO_Pregunta'])} – {top_q['Pregunta']} ({int(top_q['respuestas'])} respuestas)")

            # Grafico de barras para esa pregunta por opción
            q_rows = df_section[df_section["NO_Pregunta"] == top_q["NO_Pregunta"]]
            bar = (
                q_rows.groupby("Opcion", dropna=False)
                .size()
                .reset_index(name="conteo")
                .sort_values("conteo", ascending=False)
                .head(10)
            )
            if not bar.empty:
                fig = plt.figure()
                plt.bar(bar["Opcion"].astype(str), bar["conteo"])
                plt.xticks(rotation=45, ha="right")
                plt.title("Top opciones en la pregunta más respondida")
                plt.tight_layout()
                st.pyplot(fig)

    # Ejemplo de corte demográfico (si hay columnas estándares)
    demo_cols = [c for c in ["género", "genero", "sexo", "edad", "region", "región"] if c in df_section.columns]
    if demo_cols:
        demo = demo_cols[0]  # primer demográfico disponible
        st.markdown(f"**Distribución por {demo}:**")
        dist = (
            df_section.groupby(demo, dropna=False)
            .size()
            .reset_index(name="conteo")
            .sort_values("conteo", ascending=False)
        )
        st.dataframe(dist, use_container_width=True)

# ------------------
# UI
# ------------------

st.title("💬 Chatbot Analítico – Paciente Digital 2025")

st.markdown(
    "Este asistente analiza los resultados del estudio **Paciente Digital 2025**. "
    "Escribe **hola** para comenzar o haz una pregunta directamente."
)


# Carga de archivos: permitir tanto carga local como desde el directorio actual
with st.expander("📁 Cargar/usar archivos CSV"):
    col1, col2 = st.columns(2)
    with col1:
        up_preguntas = st.file_uploader("Preguntas_Secciones_Publico.csv", type=["csv"])
        up_resultados = st.file_uploader("Resultados.csv", type=["csv"])
    with col2:
        up_demograficos = st.file_uploader("Demograficos.csv", type=["csv"])
        up_padecimientos = st.file_uploader("Padecimientos_Homologados.csv", type=["csv"])

# Intentar cargar archivos (subidos o locales)
preguntas = try_load_local_or_upload("Preguntas_Secciones_Publico.csv", up_preguntas)
resultados = try_load_local_or_upload("Resultados.csv", up_resultados)
demograficos = try_load_local_or_upload("Demograficos.csv", up_demograficos)
padecimientos = try_load_local_or_upload("Padecimientos_Homologados.csv", up_padecimientos)

# Secciones desde el archivo de preguntas
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial del chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input del usuario
user_input = st.chat_input("Escribe un mensaje...")

# ===============================
# Flujo principal
# ===============================

if user_input:
    # Mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Crear DataFrame unificado
    df_unificado = build_unified_df(resultados, preguntas, demograficos, padecimientos)
    llm = azure_llm_from_env(temperature=0.2)
    agent = make_agent(df_unificado, llm) if not df_unificado.empty else None
    response = ""

    # ============================================
    # Caso 1: Saludo inicial
    # ============================================
    if user_input.strip().lower() in ["hola", "buenas", "hi", "hello"]:
        secciones = get_sections_sorted(preguntas)
        if secciones:
            lista_opciones = "\n".join([f"{i+1}) **{s}**" for i, s in enumerate(secciones)])
            response = (
                "👋 ¡Hola! Soy tu asistente para el estudio **Paciente Digital 2025**.\n\n"
                "Estas son las secciones disponibles:\n\n"
                f"{lista_opciones}\n\n"
                f"{len(secciones)+1}) **Hacer una pregunta concreta**"
            )
        else:
            response = (
                "👋 ¡Hola! No encontré secciones disponibles. "
                "Asegúrate de que `Preguntas_Secciones_Publico.csv` tenga la columna `Seccion`."
            )

    # ===============================
    # Caso 2: Selección de sección o pregunta
    # ===============================
    else:
        seleccion = user_input.strip()
        handled = False

        # ------------------------------------------------------------------
        # 0) MENÚ PRINCIPAL: si NO hay sección activa, un dígito => SECCIÓN
        # ------------------------------------------------------------------
        if not st.session_state.get("current_section"):
            secciones = get_sections_sorted(preguntas)

            if seleccion.isdigit():
                idx = int(seleccion)

                # Opción "Hacer una pregunta concreta" (= última + 1)
                if idx == len(secciones) + 1:
                    # Limpia cualquier rastro de filtros/selecciones anteriores
                    for k in [
                        "current_questions", "current_question", "current_question_text",
                        "filter_mode", "awaiting_filter_value", "current_filters",
                        "padecimientos_list", "awaiting", "last_table_payload",
                        "estados_list", "generos_list", "awaiting_llm_question"
                    ]:
                        st.session_state.pop(k, None)

                    with st.chat_message("assistant"):
                        st.markdown(
                            "Claro, puedes hacerme una pregunta sobre los resultados del estudio.\n\n"
                            "**Ejemplos:**\n"
                            "- ¿Qué dispositivo usan más las mujeres?\n"
                            "- ¿Cuál es el porcentaje de personas con diabetes que usan smartwatch?\n"
                            "- ¿Qué canal digital prefieren los adultos mayores?\n\n"
                            "**Escribe tu pregunta a continuación:**"
                        )
                    st.session_state["awaiting_llm_question"] = True
                    handled = True
                    st.stop()

                # Selección de sección válida
                elif 1 <= idx <= len(secciones):
                    seccion = secciones[idx - 1]

                    # 🔄 Limpia SIEMPRE todo el estado de filtros y selección
                    for k in [
                        "current_question", "current_question_text", "filter_mode",
                        "awaiting_filter_value", "current_filters", "padecimientos_list",
                        "awaiting", "awaiting_llm_question", "last_table_payload",
                        "estados_list", "generos_list"
                    ]:
                        st.session_state.pop(k, None)

                    # Guarda sección actual y construye lista de preguntas
                    st.session_state["current_section"] = seccion
                    preguntas_seccion = preguntas[preguntas["Seccion"] == seccion].copy()
                    if "NO_Pregunta" in preguntas_seccion.columns:
                        preguntas_seccion = preguntas_seccion.sort_values("NO_Pregunta")

                    st.session_state["current_questions"] = [
                        (
                            int(row["NO_Pregunta"]) if "NO_Pregunta" in row and pd.notna(row["NO_Pregunta"]) else None,
                            str(row["Pregunta"]),
                        )
                        for _, row in preguntas_seccion.iterrows()
                    ]

                    # Muestra el listado de preguntas de la sección
                    if st.session_state["current_questions"]:
                        lines = []
                        for i, (no_p, texto) in enumerate(st.session_state["current_questions"], start=1):
                            label = f"{no_p}. " if no_p is not None else ""
                            lines.append(f"{i}. {label}{texto}")
                        lista_preguntas = "\n".join(lines)

                        response = (
                            f"### 📊 Preguntas disponibles en **{seccion}**:\n\n"
                            f"{lista_preguntas}\n\n"
                            f"{len(st.session_state['current_questions'])+1}. 💬 Hacer una pregunta concreta\n"
                            f"{len(st.session_state['current_questions'])+2}. 🔙 Volver al menú anterior\n\n"
                            f"_Selecciona una opción escribiendo su número (p. ej. `2`)._"
                        )
                    else:
                        response = f"No se encontraron preguntas para la sección **{seccion}**."

                    with st.chat_message("assistant"):
                        st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.stop()

            # Si no digitó número en menú principal, deja seguir (puede ser texto/LLM)
            # pero no marques handled aquí.

        # ------------------------------------------------------------------
        # 1) YA HAY SECCIÓN ACTIVA: número => PREGUNTA / ACCIÓN / FILTROS
        # ------------------------------------------------------------------
        if st.session_state.get("current_section") and st.session_state.get("current_questions"):

            # Si venimos del menú post-resultados
            if st.session_state.get("awaiting") == "post_results_menu" and seleccion.isdigit():
                idx = int(seleccion)

                # 1) Volver al listado de preguntas
                if idx == 1:
                    st.session_state.pop("current_question", None)
                    st.session_state.pop("current_question_text", None)
                    st.session_state.pop("awaiting", None)

                    seccion = st.session_state["current_section"]
                    preguntas_seccion = preguntas[preguntas["Seccion"] == seccion].copy()
                    if "NO_Pregunta" in preguntas_seccion.columns:
                        preguntas_seccion = preguntas_seccion.sort_values("NO_Pregunta")

                    st.session_state["current_questions"] = [
                        (
                            int(row["NO_Pregunta"]) if "NO_Pregunta" in row and pd.notna(row["NO_Pregunta"]) else None,
                            str(row["Pregunta"]),
                        )
                        for _, row in preguntas_seccion.iterrows()
                    ]

                    qs = st.session_state["current_questions"]
                    lines = [f"{i}. {limpiar_pregunta(texto)}" for i, (_, texto) in enumerate(qs, start=1)]
                    lines.append(f"{len(lines)+1}. 💬 Hacer una pregunta concreta")
                    lines.append(f"{len(lines)+2}. 🔙 Volver al menú anterior")
                    lista_preguntas = "\n".join(lines)

                    response = (
                        f"### 📊 Preguntas disponibles en **{seccion}**:\n\n"
                        f"{lista_preguntas}\n\n"
                        "_Selecciona una opción escribiendo su número._"
                    )

                    with st.chat_message("assistant"):
                        st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.stop()

                # 3) Filtrar por Padecimiento
                elif idx == 3 and "Padecimiento Homologado" in padecimientos.columns:
                    padecimientos_list = sorted(padecimientos["Padecimiento Homologado"].dropna().unique().tolist())
                    st.session_state["padecimientos_list"] = padecimientos_list

                    lista_pads = "\n".join([f"{i+1}. {p}" for i, p in enumerate(padecimientos_list)])
                    response = (
                        f"### 🩺 Padecimientos disponibles (elige una opción):\n\n"
                        f"{lista_pads}\n\n"
                        "_Escribe el número correspondiente al padecimiento._"
                    )

                    with st.chat_message("assistant"):
                        st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state["awaiting"] = "filter_padecimiento"
                    st.stop()

                # 4) Filtrar por Estado
                elif idx == 4 and "Estado" in demograficos.columns:
                    estados_list = sorted(demograficos["Estado"].dropna().unique().tolist())
                    st.session_state["estados_list"] = estados_list

                    lista_est = "\n".join([f"{i+1}. {e}" for i, e in enumerate(estados_list)])
                    response = (
                        f"### 🗺️ Estados disponibles (elige una opción):\n\n"
                        f"{lista_est}\n\n"
                        "_Escribe el número correspondiente al estado._"
                    )

                    with st.chat_message("assistant"):
                        st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state["awaiting"] = "filter_estado"
                    st.stop()

                # 5) Filtrar por Género
                elif idx == 5 and "Genero" in demograficos.columns:
                    generos_list = sorted(demograficos["Genero"].dropna().unique().tolist())
                    st.session_state["generos_list"] = generos_list

                    lista_gen = "\n".join([f"{i+1}. {g}" for i, g in enumerate(generos_list)])
                    response = (
                        f"### 🚻 Géneros disponibles (elige una opción):\n\n"
                        f"{lista_gen}\n\n"
                        "_Escribe el número correspondiente al género._"
                    )

                    with st.chat_message("assistant"):
                        st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state["awaiting"] = "filter_genero"
                    st.stop()

            # Elección de padecimiento (tras listar)
            elif st.session_state.get("awaiting") == "filter_padecimiento" and seleccion.isdigit():
                idx = int(seleccion)
                padecimientos_list = st.session_state.get("padecimientos_list", [])
                if 1 <= idx <= len(padecimientos_list):
                    padecimiento_sel = padecimientos_list[idx - 1]
                    no_pregunta = st.session_state["current_question"]
                    texto_preg = st.session_state["current_question_text"]

                    ids_padec = padecimientos[padecimientos["Padecimiento Homologado"] == padecimiento_sel]["respondent_id"].unique()
                    df_q = resultados[resultados["NO_Pregunta"] == no_pregunta]
                    df_filtro = df_q[df_q["respondent_id"].isin(ids_padec)]

                    if df_filtro.empty:
                        response = f"⚠️ No se encontraron respuestas para **{padecimiento_sel}**."
                    else:
                        n_filtro = df_filtro["respondent_id"].nunique()
                        tabla_f = (
                            df_filtro.groupby("Opcion", dropna=False)["respondent_id"]
                            .nunique()
                            .reset_index(name="conteo")
                            .sort_values("conteo", ascending=False)
                        )
                        tabla_f["porcentaje"] = (tabla_f["conteo"] / n_filtro * 100).round(1)
                        tabla_md = "| Opción de respuesta | % de menciones |\n|----------------------|----------------|\n"
                        for _, row in tabla_f.iterrows():
                            tabla_md += f"| {row['Opcion']} | {row['porcentaje']}% |\n"

                        top_opcion = tabla_f.iloc[0]["Opcion"]
                        top_pct = tabla_f.iloc[0]["porcentaje"]

                        response = (
                            f"### 📊 Resultados (Filtro aplicado: **{padecimiento_sel}**)\n\n"
                            f"**Pregunta:** *{texto_preg}*\n\n"
                            f"{tabla_md}\n\n"
                            f"**n = {n_filtro} (respondientes únicos con {padecimiento_sel})**\n\n"
                            f"🧠 *Interpretación:* La opción más mencionada fue **'{top_opcion}'** "
                            f"con un {top_pct}% de los respondientes.\n\n"
                            f"📘 *Fuente: Encuesta Paciente Digital 2025*\n\n"
                            f"---\n"
                            f"**Opciones:**\n"
                            f"1) 🔙 Volver al menú anterior\n"
                            f"2) 💬 Hacer una pregunta concreta\n"
                            f"3) 🩺 Filtrar por Padecimiento (cambiar)\n"
                            f"4) 🗺️ Filtrar por Estado\n"
                            f"5) 🚻 Filtrar por Género"
                        )

                    with st.chat_message("assistant"):
                        st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state["awaiting"] = "post_results_menu"
                    st.stop()

            # Listado de preguntas (sin resultado aún)
            else:
                qs = st.session_state["current_questions"]
                n = len(qs)

                if seleccion.isdigit():
                    idx = int(seleccion)

                    # --- Volver al menú principal ---
                    volver_idx = n + 2
                    if idx == volver_idx:
                        for key in [
                            "current_section", "current_questions", "current_question",
                            "current_question_text", "padecimientos_list", "awaiting",
                            "estados_list", "generos_list", "last_table_payload"
                        ]:
                            st.session_state.pop(key, None)

                        secciones = get_sections_sorted(preguntas)
                        lista_opciones = "\n".join([f"{i+1}) **{s}**" for i, s in enumerate(secciones)])
                        response = (
                            "🏠 **Has vuelto al menú principal.**\n\n"
                            "Estas son las secciones disponibles:\n\n"
                            f"{lista_opciones}\n\n"
                            f"{len(secciones)+1}) **Hacer una pregunta concreta**"
                        )

                        with st.chat_message("assistant"):
                            st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.stop()

                    # --- Hacer una pregunta concreta dentro de la sección ---
                    if idx == n + 1:
                        for k in [
                            "current_question", "current_question_text", "filter_mode",
                            "awaiting_filter_value", "current_filters", "padecimientos_list",
                            "awaiting", "last_table_payload"
                        ]:
                            st.session_state.pop(k, None)

                        with st.chat_message("assistant"):
                            st.markdown("Perfecto. **Escribe tu pregunta concreta** sobre esta sección o el estudio.")
                        st.session_state["awaiting_llm_question"] = True
                        st.stop()

                    # --- Elegir pregunta ---
                    if 1 <= idx <= n:
                        no_pregunta, texto_preg = qs[idx - 1]
                        st.session_state["current_question"] = no_pregunta
                        st.session_state["current_question_text"] = texto_preg

                        df_q = resultados[resultados["NO_Pregunta"] == no_pregunta]
                        if not df_q.empty:
                            n_muestra = df_q["respondent_id"].nunique()
                            tabla = (
                                df_q.groupby("Opcion", dropna=False)["respondent_id"]
                                .nunique()
                                .reset_index(name="conteo")
                                .sort_values("conteo", ascending=False)
                            )
                            tabla["porcentaje"] = (tabla["conteo"] / n_muestra * 100).round(1)
                            tabla_md = "| Opción de respuesta | % de menciones |\n|----------------------|----------------|\n"
                            for _, row in tabla.iterrows():
                                tabla_md += f"| {row['Opcion']} | {row['porcentaje']}% |\n"

                            top_opcion = tabla.iloc[0]["Opcion"]
                            top_pct = tabla.iloc[0]["porcentaje"]

                            response = (
                                f"### 📊 Pregunta: *{texto_preg}*\n\n"
                                f"**Resultados (sin filtro):**\n\n{tabla_md}\n\n"
                                f"**n = {n_muestra} (respondientes únicos)**\n\n"
                                f"🧠 *Interpretación:* La opción más seleccionada fue **'{top_opcion}'** "
                                f"con un {top_pct}% de los respondientes.\n\n"
                                f"📘 *Fuente: Encuesta Paciente Digital 2025*\n\n"
                                f"---\n"
                                f"**Opciones:**\n"
                                f"1) 🔙 Volver al menú anterior\n"
                                f"2) 💬 Hacer una pregunta concreta\n"
                                f"3) 🩺 Filtrar por Padecimiento\n"
                                f"4) 🗺️ Filtrar por Estado\n"
                                f"5) 🚻 Filtrar por Género"
                            )

                            st.session_state["awaiting"] = "post_results_menu"

                        with st.chat_message("assistant"):
                            st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.stop()

        # ------------------------------------------------------------------
        # 2) Si NO hubo sección ni pregunta resuelta, pasa al LLM (si corresponde)
        # ------------------------------------------------------------------
        if not handled and agent is not None:
            instruct = ChatPromptTemplate.from_template(
                """
                Eres un analista de investigación del estudio "Paciente Digital 2025".
                Tu objetivo es interpretar, comparar y contextualizar resultados de encuestas.
                Responde con claridad, pasos reproducibles y, cuando corresponda, incluye porcentajes o totales.
                Describe qué columnas del DataFrame usas, qué filtros aplicas y cómo obtienes las conclusiones.
                Si la pregunta del usuario no es clara o no hay datos suficientes, pide más contexto educadamente.

                Pregunta del usuario: {q}
                """
            )
            try:
                full_q = instruct.format(q=user_input)
                result = agent.invoke({"input": full_q})
                response = result.get("output", str(result)) if isinstance(result, dict) else str(result)
            except Exception as e:
                if "Could not parse LLM output" in str(e):
                    response = (
                        "⚠️ No pude interpretar tu pregunta. "
                        "Por favor sé más específico o intenta algo como: "
                        "`porcentaje por género en la pregunta 26`."
                    )
                else:
                    response = f"Ocurrió un error al analizar la pregunta: {e}"

            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


        # ==========================================================
        # Caso 3: Hacer una pregunta concreta (usa el LLM)
        # ==========================================================
        if seccion == "Hacer una pregunta concreta":
            # El usuario eligió la opción 7
            st.session_state["awaiting"] = "user_question"

            response = (
                "🧠 Claro, puedes hacerme una pregunta sobre los resultados del estudio.\n\n"
                "Por ejemplo:\n"
                "- ¿Qué dispositivo usan más las mujeres?\n"
                "- ¿Cuál es el porcentaje de personas con diabetes que usan smartwatch?\n"
                "- ¿Qué canal digital prefieren los adultos mayores?\n\n"
                "✍️ Escribe tu pregunta a continuación:"
            )

            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.stop()

        # ==========================================================
        # Caso 4: Si el usuario está en modo 'awaiting question'
        # ==========================================================
        if st.session_state.get("awaiting") == "user_question":
            if user_input.strip() and user_input.strip() not in ["7"]:
                if agent is not None:
                    with st.chat_message("assistant"):
                        st.markdown("🤖 Analizando tu pregunta con IA...")
                    try:
                        respuesta_llm = agent.run(user_input)
                        response = f"### 🤖 Respuesta generada por IA:\n\n{respuesta_llm}"
                        response += (
                            "\n\n---\n\n"
                            "### 🧭 Opciones:\n"
                            "1️⃣ **Volver al menú anterior**  \n"
                            "2️⃣ **Hacer una pregunta concreta**  \n"
                            "3️⃣ **Filtrar por Padecimiento**  \n"
                            "4️⃣ **Filtrar por Estado**  \n"
                            "5️⃣ **Filtrar por Género**  \n\n"
                            "_Selecciona una opción escribiendo su número._"
                        )
                    except Exception as e:
                        if "Could not parse LLM output" in str(e):
                            response = (
                                "⚠️ No pude interpretar tu pregunta. "
                                "Por favor intenta algo como: "
                                "`porcentaje de hombres que usan smartwatch`."
                            )
                        else:
                            response = f"⚠️ Error al ejecutar el agente: {e}"
                else:
                    response = "⚠️ No se pudo crear el agente. Verifica los archivos CSV cargados."

                st.session_state["awaiting"] = None
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.stop()
        
        # ==========================================================
        # Manejar opciones después de una respuesta del LLM
        # ==========================================================
        if st.session_state.get("awaiting") is None and user_input.strip().isdigit():
            opcion = int(user_input.strip())

            # --- Opción 1: Volver al menú principal ---
            if opcion == 1:
                for key in ["current_section", "current_questions", "current_question",
                            "current_question_text", "padecimientos_list", "awaiting"]:
                    st.session_state.pop(key, None)

                secciones = get_sections_sorted(preguntas)
                lista_opciones = "\n".join([f"{i+1}) **{s}**" for i, s in enumerate(secciones)])
                response = (
                    "🏠 **Has vuelto al menú principal.**\n\n"
                    "Estas son las secciones disponibles:\n\n"
                    f"{lista_opciones}\n\n"
                    f"{len(secciones)+1}) **Hacer una pregunta concreta**"
                )

                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.stop()

            # --- Opción 2: Hacer una nueva pregunta concreta ---
            elif opcion == 2:
                st.session_state["awaiting"] = "user_question"
                response = (
                    "🧠 Perfecto, puedes hacerme otra pregunta sobre los resultados.\n\n"
                    "Por ejemplo:\n"
                    "- ¿Qué dispositivo usan más las mujeres?\n"
                    "- ¿Cuál es el porcentaje de personas con diabetes que usan smartwatch?\n"
                    "- ¿Qué canal digital prefieren los adultos mayores?\n\n"
                    "✍️ Escribe tu nueva pregunta:"
                )

                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.stop()

            # --- Opción 3: Filtrar por Padecimiento ---
            elif opcion == 3:
                if "Padecimiento Homologado" not in padecimientos.columns:
                    response = "⚠️ No hay datos de padecimientos cargados."
                else:
                    st.session_state["awaiting"] = "filter_padecimiento"
                    padecimientos_list = sorted(padecimientos["Padecimiento Homologado"].dropna().unique())
                    st.session_state["padecimientos_list"] = padecimientos_list

                    lista_pads = "\n".join([f"{i+1}. {p}" for i, p in enumerate(padecimientos_list)])

                    response = (
                        "### 🩺 Filtrar por Padecimiento\n\n"
                        "Selecciona uno de los siguientes padecimientos:\n\n"
                        f"{lista_pads}\n\n"
                        "_Escribe el número correspondiente._"
                    )

                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.stop()

            # --- Opción 4: Filtrar por Estado ---
            elif opcion == 4:
                if "Estado" not in demograficos.columns:
                    response = "⚠️ No hay información de Estados en el dataset."
                else:
                    st.session_state["awaiting"] = "filter_estado"
                    estados_list = sorted(demograficos["Estado"].dropna().unique())
                    st.session_state["estados_list"] = estados_list

                    lista_est = "\n".join([f"{i+1}. {e}" for i, e in enumerate(estados_list)])

                    response = (
                        "### 🗺️ Filtrar por Estado\n\n"
                        f"{lista_est}\n\n"
                        "_Escribe el número correspondiente._"
                    )

                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.stop()

            # --- Opción 5: Filtrar por Género ---
            elif opcion == 5:
                if "Genero" not in demograficos.columns:
                    response = "⚠️ No hay datos de género cargados."
                else:
                    st.session_state["awaiting"] = "filter_genero"
                    generos_list = sorted(demograficos["Genero"].dropna().unique())
                    st.session_state["generos_list"] = generos_list

                    lista_gen = "\n".join([f"{i+1}. {g}" for i, g in enumerate(generos_list)])

                    response = (
                        "### 🚻 Filtrar por Género\n\n"
                        f"{lista_gen}\n\n"
                        "_Escribe el número del género._"
                    )

                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.stop()




    # ============================================
    # Mostrar respuesta final
    # ============================================
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
