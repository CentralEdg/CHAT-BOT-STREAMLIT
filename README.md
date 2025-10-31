
# Chatbot Analítico – Paciente Digital 2025

Este proyecto crea un chatbot con Streamlit + LangChain (Azure OpenAI) para interpretar y comparar resultados de una encuesta.
Cumple con el flujo de **Inicio del Chat** descrito: saludo, explicación y listado **exacto** de secciones (alfabético, numerado y con opción final "Hacer una pregunta concreta").

## Archivos esperados
- `Preguntas_Secciones_Publico.csv`
- `Resultados.csv`
- `Demograficos.csv`
- `Padecimientos_Homologados.csv`

Puedes subirlos desde la UI de la app o colocarlos en el mismo directorio donde ejecutes `streamlit run app.py`.

## Variables de entorno
Crea un archivo `.env` con:

```
AZURE_OPENAI_API_KEY=YOUR_KEY
AZURE_OPENAI_ENDPOINT=https://an-report.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

## Instalación
```bash
python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución
```bash
streamlit run app.py
```

## Notas
- El listado de secciones proviene **únicamente** del CSV de preguntas y se muestra en orden alfabético, sin traducir ni reformular.
- Para preguntas concretas, el agente analiza el DataFrame unificado (joins por `respondent_id` y `NO_Pregunta`).
- Si alguna columna demográfica tiene acentos diferentes (p. ej. "género"/"genero"), la app intentará identificar al menos una para mostrar una distribución básica.
