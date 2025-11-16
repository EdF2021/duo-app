# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import os
import logging
import time
import json
from io import BytesIO
from typing_extensions import override

import pandas as pd
import streamlit as st
from openai import OpenAI, AssistantEventHandler
from streamlit_extras.bottom_container import bottom
from PIL import Image
from streamlit_mic_recorder import mic_recorder

# -----------------------------------------------------------------------------
# CONSTANTEN EN VARIABELEN
# -----------------------------------------------------------------------------
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
teller = 0
# It's recommended to set the API key via environment variables or st.secrets
# For example: client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
client = OpenAI()

ASSISTANT_ID = "asst_35ww7mbISi8fro7aqH8ZvL8l"  # De id van De Toetser.
VECTOR_ID = "vs_JaqgO059Cc7qFRFKbW5ZyXa1"

# Padbeheer
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = "images/sync.jpeg"  # os.path.join(BASE_DIR, "images", "sync.jpeg")
LOG_PATH = "log.txt"

image = Image.open(IMAGE_PATH)


# -----------------------------------------------------------------------------
# HULP FUNCTIES
# -----------------------------------------------------------------------------
def show_json(obj):
    """Toon de json output van de assistant for debugging."""
    try:
        # Pretty print the JSON for better readability in logs
        LOGGER.info(json.dumps(obj.model_dump(), indent=2))
    except Exception as e:
        LOGGER.error(f"Could not serialize object to JSON: {e}")
        LOGGER.info(obj)


# -----------------------------------------------------------------------------
# CONFIGURATIE VAN DE PAGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="De Data Democraat",
    page_icon="🧑‍💻",
    layout="centered",
    initial_sidebar_state="collapsed",
)
col1, col2 = st.columns([1, 1])
with col1:
    st.header(
        "🧑‍💻 :blue[**De Data Democraat**]\n**Waarom zou je moeilijk doen ....?** "
    )
    with st.expander(" :bulb: :blue[**Maak het jezelf makkelijk en werk slim!**]"):
        st.markdown(
            """##### Upload een databestand en de Data Democraat begint met het uitvoeren van de volgende 3 stappen:
1. **Data opschonen**: Verwijderen van nulwaarden, irrelevante kolommen en rijen en transformeren van de data.
2. **Data analyseren**: Berekenen van beschrijvende statistieken en zoeken naar opvallende zaken.
3. **Data visualiseren**: Genereren van grafieken en plots om de bevindingen te visualiseren.

**Chat met de assistent**: Stel vragen aan de AI-assistent voor verdere inzichten en aanbevelingen.
"""
        )

        st.markdown(
            """ **De Data Democraat helpt bij het analyseren van data.**                
            
         
        """
        )

with col2:
    st.image(
        image,
        caption=None,
        width=320,
        clamp=True,
        channels="RGB",
        output_format="auto",
    )


# -----------------------------------------------------------------------------
# INITIALISATIES
# -----------------------------------------------------------------------------
def log_activity():
    """Registreer activiteiten in een logbestand."""
    if not st.session_state.get("is_in_onderzoek_logboek_data"):
        st.session_state["is_in_onderzoek_logboek_data"] = True
        log_message = f"\nDe Data Democraat geopend op {time.asctime()}\n"
        try:
            with open(LOG_PATH, "a") as f:
                f.write(log_message)
            LOGGER.info("Activity logged.")
        except Exception as e:
            LOGGER.error(f"Failed to write to log.txt: {e}")


def set_model():
    """Stel het model in op basis van authenticatie status."""
    if "authentication_status" not in st.session_state:
        return "gpt-4o-mini"
    return "gpt-4o" if st.session_state.get("authentication_status") else "gpt-4o-mini"


def init_session_states():
    """Initialiseer de sessievariabelen."""
    if "is_initiated" not in st.session_state:
        st.session_state.is_initiated = True
        st.session_state.openai_model = set_model()
        st.session_state.assistant_id = ASSISTANT_ID
        st.session_state.vector_id = VECTOR_ID
        st.session_state.file_ids = []
        st.session_state.thread_id = None
        st.session_state.run_id = None
        st.session_state.messages = []
        st.session_state.last_message_id = None
        st.session_state.processing = False
        st.session_state.data_list = []
        LOGGER.info("Session state initialized.")


init_session_states()

with st.sidebar:
    if st.button("Verwijder Geschiedenis"):
        # Reset relevant session state variables
        keys_to_reset = [
            "thread_id",
            "file_ids",
            "messages",
            "data_list",
            "run_id",
            "last_message_id",
        ]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# -----------------------------------------------------------------------------
# DATA INLEZEN
# -----------------------------------------------------------------------------
def _load_dataframe_from_source(source):
    """Laadt een dataframe van een bron (URL of UploadedFile)."""
    df, file_name = None, None
    try:
        if isinstance(source, str):  # Het is een URL
            file_name = source.split("/")[-1] or "data_from_url"
        else:  # Het is een UploadedFile
            file_name = source.name

        file_extension = os.path.splitext(file_name)[1].lower()

        if file_extension == ".csv":
            df = pd.read_csv(source)
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(source)
        else:
            st.warning(
                f"Onbekende bestandsextensie '{file_extension}' voor {file_name}. Poging tot lezen als CSV."
            )
            df = pd.read_csv(source)
            if isinstance(source, str):
                file_name += ".csv"
    except Exception as e:
        st.error(f"Fout bij het lezen van bron {file_name}: {e}")
        return None, None
    return df, file_name


def get_file():
    """Inlezen van data via bestandsupload of URL, omzetten naar een dataframe en uploaden naar Assistant."""
    input_method = st.radio(
        "Kies de invoermethode:",
        ("Bestand uploaden", "URL"),
        horizontal=True,
    )
    st.session_state.input_method = input_method

    sources = []
    if input_method == "Bestand uploaden":
        uploaded_files = st.file_uploader(
            " :point_right: :blue[**UPLOAD HIER HET DATABESTAND (excel of csv)**] ",
            type=["xlsx", "csv"],
            help="Op dit moment kun je alleen xlsx en csv bestanden uploaden!",
            accept_multiple_files=True,
        )
        if not uploaded_files:
            st.warning("Wachten op een databestand om analyse te starten...")
            st.stop()
        sources.extend(uploaded_files)
    elif input_method == "URL":
        url_input = st.text_input("Voer de URL van het databestand in (CSV of Excel):")
        if not url_input:
            st.warning("Voer een URL in om de analyse te starten...")
            st.stop()
        sources.append(url_input)

    data_list, file_ids = [], []
    for source in sources:
        df, file_name = _load_dataframe_from_source(source)
        if df is None:
            continue

        data_list.append(df)
        st.dataframe(df.head(), use_container_width=True)

        try:
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            file_object = client.files.create(
                file=(f"{os.path.splitext(file_name)[0]}.csv", csv_buffer),
                purpose="assistants",
            )
            file_ids.append(file_object.id)
            LOGGER.info(
                f"Bron {file_name} verwerkt en geüpload met file_id: {file_object.id}"
            )
        except Exception as e:
            st.error(f"Fout bij het uploaden van bestand naar OpenAI: {e}")
            st.stop()

    if not data_list:
        st.warning("Geen data ingelezen. Selecteer een bestand of geef een URL op.")
        st.stop()

    return data_list, file_ids


# Inlezen van de DataFrame en file toevoegen aan de Assistent
data_list, file_ids = get_file()
st.session_state.data_list = data_list
st.session_state.file_ids = file_ids


# -----------------------------------------------------------------------------
# CHAT EN ASSISTANT FUNCTIES
# -----------------------------------------------------------------------------
class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        st.session_state.current_response += "\n\n"
        st.session_state.placeholder.markdown(st.session_state.current_response)

    @override
    def on_text_delta(self, delta, snapshot):
        st.session_state.current_response += delta.value
        st.session_state.placeholder.markdown(st.session_state.current_response + "▌")

    @override
    def on_tool_call_created(self, tool_call):
        pass

    @override
    def on_tool_call_delta(self, delta, snapshot):
        pass


def stream_and_run(thread_id, assistant_id):
    """Stream the assistant's response."""
    try:
        with client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant_id,
            event_handler=EventHandler(),
        ) as stream:
            stream.until_done()
    except Exception as e:
        st.error(f"Er is een fout opgetreden tijdens het streamen: {e}")
        LOGGER.error(f"Streaming error: {e}")


def convert_file_to_png(file_id):
    """Helper function to convert an output file from OpenAI to a displayable PNG."""
    try:
        data = client.files.content(file_id)
        data_bytes = data.read()
        st.image(data_bytes, output_format="PNG")
        LOGGER.info(f"IMAGE OPGESLAGEN EN GETOOND: {file_id}")
    except Exception as e:
        st.error(f"Kon afbeelding niet weergeven: {e}")
        LOGGER.error(f"Error converting file {file_id} to png: {e}")


def get_report(file_id):
    """Download a report file from OpenAI and provide a download button."""
    try:
        data_bytes = client.files.content(file_id).read()
        st.download_button(
            label=":red[DOWNLOAD HET RAPPORT]",
            data=data_bytes,
            file_name="report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key=f"download_{file_id}",
        )
    except Exception as e:
        st.error(f"Kon rapport niet downloaden: {e}")
        LOGGER.error(f"Error downloading report {file_id}: {e}")


def display_messages():
    """Display existing messages in the chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        role = "user" if msg.role == "user" else "assistant"
        with st.chat_message(role):
            for content_part in msg.content:
                if content_part.type == "text":
                    message_text = content_part.text.value
                    annotations = content_part.text.annotations
                    if annotations:
                        for annotation in annotations:
                            if annotation.type == "file_path":
                                file_id = annotation.file_path.file_id
                                # The assistant will often cite the file path, so we can hide it.
                                message_text = message_text.replace(annotation.text, "")
                                get_report(file_id)

                    st.markdown(message_text)
                elif content_part.type == "image_file":
                    convert_file_to_png(content_part.image_file.file_id)


def process_chat_input(prompt: str):
    """Process user input, run the assistant, and display the response."""
    if not st.session_state.thread_id:
        st.error("Sessie is niet correct gestart. Herlaad de pagina.")
        return

    # Add user message to thread
    client.beta.threads.messages.create(
        thread_id=st.session_state.thread_id,
        role="user",
        content=prompt + " Visualiseer je resultaten",
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st.session_state.placeholder = st.empty()
        st.session_state.current_response = ""
        stream_and_run(st.session_state.thread_id, st.session_state.assistant_id)
        st.session_state.placeholder.markdown(st.session_state.current_response)

    # Fetch only new messages and update history
    try:
        new_messages = client.beta.threads.messages.list(
            thread_id=st.session_state.thread_id,
            order="asc",
            after=st.session_state.get("last_message_id"),
        )

        new_message_data = new_messages.data
        if new_message_data:
            st.session_state.messages.extend(new_message_data)
            st.session_state.last_message_id = new_message_data[-1].id

        # We need to rerun to let display_messages render the new messages
        # that were just added to the state.
        st.rerun()
    except Exception as e:
        st.error(f"Kon de berichtengeschiedenis niet bijwerken: {e}")
        LOGGER.error(f"Failed to update message history: {e}")


# --- Main App Logic ---

# Create a Thread if we don't have one
if not st.session_state.get("thread_id"):
    try:
        # Update assistant with the uploaded files
        client.beta.assistants.update(
            st.session_state.assistant_id,
            tool_resources={
                "code_interpreter": {"file_ids": st.session_state.file_ids}
            },
        )
        thread = client.beta.threads.create(
            tool_resources={"code_interpreter": {"file_ids": st.session_state.file_ids}}
        )
        st.session_state.thread_id = thread.id
        LOGGER.info(f"Nieuwe thread aangemaakt: {thread.id}")
    except Exception as e:
        st.error(f"Kon de assistent of thread niet initialiseren: {e}")
        LOGGER.error(f"Thread/Assistant initialization failed: {e}")
        st.stop()

# Display chat history
display_messages()

# Initial prompt to start the analysis
if not st.session_state.messages:
    if st.session_state.get("input_method") == "URL":
        start_prompt = "De data is ingeladen via een URL. Ga ervan uit dat de data opgeschoond is en start met de analyse."
    else:
        start_prompt = """
    Voer de volgende stappen uit:
    1.  Data opschonen: het verwijderen van nul waarden, de kolommen betekenisvolle namen geven, verwijderen van niet relevante kolommen en rijen, en indien nodig het transformeren van de datatabel. Toon de opgeschoonde tabel. 
    """
    process_chat_input(start_prompt)



    # process_chat_input(transcription.text + " Visualiseer je resultaten")

    # print(transcription.text)

    # with st.spinner("Bezig met transcriberen..."):
    #     try:
    #         audio_bio = BytesIO(audio_value['bytes'])
    #         audio_bio.name = 'audio.wav'

    #         transcript = client.audio.transcriptions.create(
    #             model="gpt-4o-transcribe",
    #             file=audio_bio.read(),
    #             response_format="text"
    #         )
    #         voice_prompt = transcript.text
    #         LOGGER.info(f"Transcribed voice input: {voice_prompt}")
    #         process_chat_input(voice_prompt)
    #     except Exception as e:
    #         st.error(f"Fout bij het transcriberen van audio: {e}")
    #         LOGGER.error(f"Error transcribing audio: {e}")

if prompt := st.chat_input("Of stel hier je vraag..."):
    process_chat_input(prompt + " Visualiseer je resultaten")

# Handle user chat input
st.markdown("---")
st.markdown("##### :studio_microphone: Spraakgestuurde vraag")
teller+=1
audio_value = st.audio_input(
    "Stel je vraag aan de Data Democraat...", key="audio_input" + str(teller)
)

# audio = mic_recorder(start_prompt="▶️ Start opname", stop_prompt="⏹️ Stop opname", key='recorder', just_once=False)

if audio_value is not None:
    audio_vraag = st.audio(audio_value)
    transcription = client.audio.transcriptions.create(
        model="gpt-4o-transcribe", file=audio_value
    )
    # with st.chat_message("user"):
    #     st.markdown(transcription.text)
    prompt = transcription.text
    if prompt:
        process_chat_input(prompt + " Visualiseer je resultaten")
    audio_value = None
    prompt = None
    transcription = None
    audio_vraag = None
    teller += 1
    

# Log activity at the end of the script run
log_activity()


with bottom():
    st.markdown(
        """
    | :copyright: :grey[EdF 2025] |  :grey[ed.de.feber@gmail.com] |
    | :------          | :------     |

    """
    )
