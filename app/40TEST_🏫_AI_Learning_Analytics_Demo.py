"""

AI Learning Analytics Demo
Demonstratie van een AI-gedreven leerplatform dat gebruik maakt 
van OpenAI's GPT-4o-mini model.

Het data gedeelte is een demo en bevat geen echte gebruikersdata.
_learning_aanbevelingen():
_learning_aanbevelingen():
    Genereer aanbevelingen voor peer learning op basis van de huidige leerstatus van de student.
    De aanbevelingen zijn gebaseerd op de huidige score, studietijd en het gekozen beroep.
    Het doel is om studenten aan te moedigen om samen te werken en elkaar te helpen bij hun leerproces.
    De aanbevelingen zijn gepersonaliseerd en gericht op het niveau van de student.
    De aanbevelingen zijn gebaseerd op de huidige score, studietijd en het gekozen beroep.
Input:
    score: De huidige score van de student.
    study_hours: Het aantal studie-uren per week.
    beroep: Het gekozen beroep van de student.
Output:
    aanbevelingen: Een lijst met aanbevelingen voor peer learning.

    # Genereer aanbevelingen op basis van de huidige leerstatus van de student
    if beroep == "Bakker":
        aanbevelingen = [
            "Werk samen met een medestudent om een nieuw recept te ontwikkelen.",
            "Organiseer een bakwedstrijd met je klasgenoten.",
            "Deel je favoriete baktechnieken met elkaar.",
        ]
    
    elif beroep == "Verkoopspecialist":
        aanbevelingen = [
            "Oefen samen met een medestudent je verkooptechnieken.",
            "Organiseer een rollenspel waarin je elkaar feedback geeft.", 
            "Deel je ervaringen met het verkopen van producten.",
        ]
GEBRUIK
Usage:
    streamlit run AI_Learning_Analytics_Demo.py

Maintainer:        Ed. de Feber e.de.feber@talland.nl
Last change:       woensdag 12 maart 2025
Version:           0.1
Created:           2025-03-08

"""

# -------------------------------------
# IMPORTS
# -------------------------------------
import re
import streamlit as st
from openai import OpenAI
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from docreader import read_file
from logger_decorator import logger
from edfutils import set_starttijd, vandaag, tijd_verschil_microsec, trunc
from streamlit_extras.bottom_container import bottom
from typing import Optional, Any, cast
import random


# ------------------------------------------------
# CONSTANTEN & INITIALISATIE
# ------------------------------------------------
client = OpenAI()
MODEL = "gpt-4o-mini"
start_tijd = set_starttijd()
vandaag_str = vandaag()
niveau_lijst = ["Starter", "Onderweg", "Gevorderde", "Expert"]
gemiddelde_groei = 0
beroepen_lijst = sorted(
    [
        "Bakker",
        "Gespecialiseerd pedagogisch medewerker",
        "Verkoopspecialist",
        "Dames kapper",
        "Monteur",
        "Chauffeur",
        "Doktersassistent",
        "Tandarts assistente",
        "Accountmanager",
        "Assistent dienstverlening",
        "Werktuigbouw",
        "Verzorgende",
        "Kok",
        "Gastheer gastvrouw",
        "Lasser",
        "Data-analist",
        "Installatie technicus",
        "Fysiotherapeut",
        "Psycholoog",
        "Leraar",
        "Pedagoog",
        "HR medewerker",
        "Sales manager",
    ]
)


# ---------------------------------------------------------------------------
# PAGINA CONFIGURATIE
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Driven Analytics Dashboard",
    page_icon="🧮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------
# Streamlit UI
# ------------------------------------------------
st.header("🧮:blue[**AI Driven Analytics Dashboard**]")
st.subheader(":blue[**Gepersonaliseerd leren met AI**]")


# ---------------------------------------------------------------------------
# TRAININGSDATA EN MODEL (gecachet)
# ---------------------------------------------------------------------------
def get_training_data():
    X = np.array(
        [
            [70, 10],
            [50, 8],
            [90, 5],
            [30, 10],
            [85, 6],
            [50, 3],
            [60, 5],
            [78, 5],
            [60, 8],
            [80, 7],
            [85, 6],
            [90, 8],
            [65, 6],
            [58, 7],
            [63, 7],
            [65, 2],
            [75, 9],
        ]
    )
    y = [
        "Onderweg",
        "Starter",
        "Expert",
        "Starter",
        "Expert",
        "Starter",
        "Onderweg",
        "Gevorderde",
        "Onderweg",
        "Gevorderde",
        "Expert",
        "Expert",
        "Onderweg",
        "Starter",
        "Starter",
        "Onderweg",
        "Gevorderde",
    ]
    return X, y


@st.cache_data
def train_model():
    X, y = get_training_data()
    model = RandomForestClassifier()
    model.fit(X, y)
    return model


model = train_model()

# ---------------------------------------------------------------------------
# SESSION STATE DEFAULTS
# ---------------------------------------------------------------------------
default_session_state = {
    "teller": 0,
    "antwoorden_van_student": [],
    "topic": None,
    "vragen": None,
    "student_essay": None,
    "feedback_response": None,
    "feedback_prompt": None,
    "score": 50,
    "old_score": 0,
    "study_hours": 5,
    "uploaded_file": None,
    "herschreven": None,
    "correcte_antwoorden": [],
    "level": "Starter",
    "toets": [],
    "feedback": None,
    "feedback_eigen_werk": None,
    "eigen_file": None,
    "opleiding": None,
    "old_opleiding": None,
    "gemiddelde_groei": 30,
    "old_gemiddelde_groei": 30,
    "naam": None,
    "groei": None,
    "oefentoets_groei": 35,
    "old_oefentoets_groei": 30,
    "messages": None,
    "bericht": None,
    "old_level": "Starter",
    "waarden": [],
    "leaderboard_sorted": {
        "Ali": 71,
        "Betty": 66,
        "Jij": 47,
        "Danielle": 31,
        "Edward": 21,
    },
}

# Initialiseer de session state variabelen
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Zodra het level wordt gewijzigd tov het oude level, of de opleiding verandert, resetten we de session states 
def reset_session_state():
    """
    Reset de relevante session state variabelen.
    """
    keys_to_reset = [
        "herschreven",
        "topic",
        "eigen_file",
        "student_essay",
        "feedback",
        "feedback_response",
        "toets",
        "antwoorden_van_student",
        "correcte_antwoorden",
        "level",
    ]
    for key in keys_to_reset:
        st.session_state[key] = (
            [] if key in ["antwoorden_van_student", "correcte_antwoorden"] else None
        )


# ---------------------------------------------------------------------------
# FUNCTIES: LEERPAD & LESMATERIAAL
# ---------------------------------------------------------------------------
def predict_learning_level(
    score: Optional[int] = None, study_hours: Optional[int] = None
) -> str:
    """
    Voorspel het leerpad van de student op basis van toetsgemiddelde en studietijd.
    """
    score = score if score is not None else st.session_state.score
    study_hours = (
        study_hours if study_hours is not None else st.session_state.study_hours
    )
    prediction = model.predict(np.array([[score, study_hours]]))[0]
    
    if prediction != st.session_state.level:
        st.session_state.old_level = st.session_state.level
        st.session_state.level = prediction
    
    # if prediction != st.session_state.old_level:
    #     reset_session_state()
    return prediction

# Doosnede van de functie
def members(L1, L2):
    """members([1,2,3],[2,3,4]) -> [2,3]"""
    return list(set([x for x in L1 for y in L2 if x == y]))

@st.fragment
@logger
def generate_lesson_content(level: str, topic: str) -> str:
    """
    Genereer uitgebreid lesmateriaal op maat.
    Dit nieuwe lesmateriaal is specifiek afgestemd op het niveau van de student.
    """
    # Geen wijzigingen in niveau en opleiding en lesmateriaal is al gegenereerd
    if (
        st.session_state.herschreven
        and st.session_state.level == level #st.session_state.old_level
        and st.session_state.topic == topic
        #and st.session_state.opleiding == st.session_state.old_opleiding
    ):
        return st.session_state.herschreven
    
    else:
        # Nieuwe opleiding, ander niveau, of lesmateriaal is nog niet gegenereerd
        st.session_state.old_gemiddelde_groei = st.session_state.gemiddelde_groei
        st.session_state.gemiddelde_groei += 3 # betrokkenheid wordt beloond
        if level != st.session_state.level:
            st.session_state.old_level = st.session_state.level
            st.session_state.level = level
        if topic != st.session_state.topic:
            st.session_state.old_topic = st.session_state.topic
            st.session_state.topic = topic
        if topic != st.session_state.opleiding:
            st.session_state.old_opleiding = st.session_state.opleiding
            st.session_state.opleiding = topic

        prompt = f"Maak uitgebreid lesmateriaal over onderwerp: {topic} op MBO {level}-niveau. Gebruik voorbeelden en oefenvragen bij ingewikkelde passages."
        response = client.chat.completions.create(
            model=MODEL,
            messages=cast(Any, [{"role": "system", "content": prompt}]),
            temperature=0.4,
        )
        st.session_state.herschreven = response.choices[0].message.content
        
        return st.session_state.herschreven if st.session_state.herschreven is not None else ""


@st.fragment
@logger
def rewrite_lesson_content(level: str, topic: str) -> str:
    """
    Herschrijf bestaand lesmateriaal zodat het past bij het niveau van de student.
    """
    if (
        st.session_state.herschreven
        and st.session_state.level == level
        and st.session_state.topic == topic
        and st.session_state.opleiding == topic
    ):
        return st.session_state.herschreven
    else:
        st.session_state.old_gemiddelde_groei = st.session_state.gemiddelde_groei
        st.session_state.gemiddelde_groei += 3  # betrokkenheid wordt beloond
        if level != st.session_state.level:
            st.session_state.old_level = st.session_state.level
            st.session_state.level = level
        if topic != st.session_state.topic:
            st.session_state.old_topic = st.session_state.topic
            st.session_state.topic = topic
        if topic != st.session_state.opleiding:
            st.session_state.old_opleiding = st.session_state.opleiding
            st.session_state.opleiding = topic

        # st.session_state.old_level = st.session_state.level
        # st.session_state.level = level
        # st.session_state.topic = topic
        # st.session_state.old_opleiding = st.session_state.opleiding
        prompt = f"""Je gaat het onderstaande lesmateriaal herschrijven zodat het geschikt is voor een MBO student op {level}-niveau.
Gebruik voorbeelden en oefenvragen bij ingewikkelde passages.

Herschrijf nu het onderstaande originele lesmateriaal:

<lesmateriaal>
{topic}
</lesmateriaal>

Maak het lesmateriaal begrijpelijker en toegankelijker voor een student op niveau {level}. Gebruik eenvoudige taal en vermijd jargon. Zorg ervoor dat de belangrijkste concepten duidelijk worden uitgelegd en dat er voldoende context wordt gegeven.

"""
        response = client.chat.completions.create(
            model=MODEL,
            messages=cast(Any, [{"role": "system", "content": prompt}]),
            temperature=0.4,
        )
        st.session_state.herschreven = response.choices[0].message.content
        
        return st.session_state.herschreven if st.session_state.herschreven is not None else ""

# file uploader vopor het uploaden van leesmateriaal
@logger
def upload_file() -> str:
    """
    Upload een lesmateriaalbestand en sla de inhoud op.
    """
    uploaded_file = st.file_uploader(
        ":blue[**Upload een lesmateriaal bestand (docx, pdf of txt bestand)**]",
        type=["docx", "pdf", "txt"],
        help="Alleen docx, pdf en txt bestanden worden ondersteund.",
        key=f"upload_{st.session_state.teller}",
    )
    if uploaded_file:
        content = read_file(uploaded_file)

        st.session_state.topic = content
        st.markdown(str(content).splitlines()[0])
        st.session_state.herschreven = None
        return str(st.session_state.topic)
    
    return str(st.session_state.topic) if st.session_state.topic is not None else ""

# file uploader voor het uploaden van eigen werk
@logger
def upload_eigen_file() -> str:
    """
    Upload een document met eigen werk en sla de inhoud op.
    """

    eigen_file = st.file_uploader(
        ":blue[**Upload je eigen werk bestand (docx, pdf of txt bestand)**]",
        type=["docx", "pdf", "txt"],
        help="Alleen docx, pdf en txt bestanden worden ondersteund.",
        key=f"eigen_file_{st.session_state.teller}",
    )
    if eigen_file:
        st.session_state.teller += 1
        content = read_file(eigen_file)
        st.session_state.old_eigen_file = st.session_state.eigen_file
        st.session_state.eigen_file = str(content)
        st.markdown(str(content).splitlines()[0])

        return str(st.session_state.eigen_file)
    return (
        str(st.session_state.eigen_file)
        if st.session_state.eigen_file is not None
        else ""
    )


@logger
def kies_beroep(beroep: str) -> str:
    """
    Selecteer een beroep uit de lijst.
    """
    if beroep != st.session_state.opleiding and beroep != "Maak een keuze":
        st.session_state.old_opleiding = st.session_state.opleiding
        st.session_state.opleiding = beroep
        return st.session_state.opleiding
    else:
        if (
            st.session_state.opleiding
            and st.session_state.opleiding != "Maak een keuze"
        ):
            return st.session_state.opleiding
        else:
            st.session_state.opleiding = "Bakker"
            return st.session_state.opleiding


@logger
def get_url(beroep: str) -> str:
    """
    Haal de beschrijving van een beroep op uit een markdown-bestand.
    """
    try:
        with open("app/data/beroepen_beschrijving.md", "r") as f:
            lines = f.read().splitlines()
        beschrijving = ""
        capture = False
        for line in lines:
            if line.strip() == f"## {beroep}":
                capture = True
            if capture:
                if line.startswith("## ") and beschrijving:
                    break
                beschrijving += line + "\n"
        return beschrijving
    except Exception as e:
        st.error(f"Fout bij het ophalen van gegevens: {e}")
        return ""


# ------------------------------------------------
# Gebruikersinvoer
# ------------------------------------------------
st.markdown(
    "###### 🪪 :red[**Vul in als simulatie koppeling studentdata & analytics engine**]"
)
col1, col2 = st.columns([1, 2])
with col1:
    naam = st.text_input("**Naam student**", "Willem Zanden")
    st.session_state.naam = naam
    beroepen = ["Maak een keuze"] + beroepen_lijst
    beroep = st.selectbox("**Opleiding**", beroepen)
    if (
        len(beroep) > 0
        and beroep != "Maak een keuze"
        and beroep != st.session_state.opleiding
    ):
        opleiding  = kies_beroep(beroep)
        if opleiding != st.session_state.opleiding:
            st.session_state.old_opleiding = st.session_state.opleiding
            st.session_state.opleiding = opleiding

            st.session_state.gemiddelde_groei = 30
            st.session_state.old_gemiddelde_groei = 20
            st.session_state.groei = None
            st.session_state.level = None
            st.session_state.old_level = None
            st.session_state.herschreven = None
            st.session_state.topic = st.session_state.opleiding
    else:
        beroep = st.text_input("**Opleiding naar keuze**", "Vul hier je opleiding in")
        if (
            beroep
            and beroep != "Maak een keuze"
            and beroep != st.session_state.opleiding
            and beroep != "Vul hier je opleiding in"
        ):
            opleiding = kies_beroep(beroep)
            if opleiding != st.session_state.opleiding:
                st.session_state.old_opleiding = st.session_state.opleiding
                st.session_state.opleiding = opleiding
                st.session_state.gemiddelde_groei = 30
                st.session_state.old_gemiddelde_groei = 20
                st.session_state.groei = None
                st.session_state.level = None
                st.session_state.old_level = None
                st.session_state.herschreven = None
                st.session_state.topic = st.session_state.opleiding

        


def change_study_hours():
    st.session_state.study_hours = st.session_state.study_hours
    return st.session_state.study_hours


def change_score():
    st.session_state.score = st.session_state.score
    return st.session_state.score


with col2:
    score = st.slider(
        "**Toetsgemiddelde in (%)**",
        min_value=10,
        max_value=100,
        value=st.session_state.score,
        on_change=change_score,
        key="score",
    )
    study_hours = st.slider(
        "**Aantal studieuren per week**",
        min_value=0,
        max_value=20,
        value=st.session_state.study_hours,
        on_change=change_study_hours,
        key="study_hours",
    )

st.divider()

# Werk de voorspelling bij als de score of studietijd verandert
if score != st.session_state.score or study_hours != st.session_state.study_hours:
    st.session_state.old_score = st.session_state.score
    st.session_state.score = score
    st.session_state.old_study_hours = st.session_state.study_hours
    st.session_state.study_hours = study_hours
    level = predict_learning_level(st.session_state.score, st.session_state.study_hours)

    if st.session_state.opleiding:
        # st.session_state.old_gemiddelde_groei = st.session_state.gemiddelde_groei
        # st.session_state.gemiddelde_groei += 3
        level = predict_learning_level(
            st.session_state.score, st.session_state.study_hours
        )
        if level != st.session_state.level:
            st.session_state.old_level = st.session_state.level
            st.session_state.level = level
            st.session_state.topic = st.session_state.opleiding
            st.session_state.old_opleiding = st.session_state.opleiding
            reset_session_state()

# Leerpad voorspellen
st.session_state.level = predict_learning_level()


def get_info_beroep(beroep: str = ""):
    """
    Haal en verwerk informatie over een beroep en genereer een groeimonitor.
    """
    if not beroep or beroep == "Maak een keuze":
        return None

    if (
        not st.session_state.groei
        or beroep != st.session_state.opleiding
        or st.session_state.level != st.session_state.old_level
    ):
        st.session_state.old_opleiding = st.session_state.opleiding
        st.session_state.opleiding = beroep

    page_content = get_url(beroep)
    prompt = f"""
<content>{page_content}</content>
Je gaat een correcte, precieze korte samenvatting maken van de content van een webpagina over het beroep {beroep}. Die samenvatting gebruik je alleen om een groeimonitor te maken. JE TOONT DE SAMENVATTING NIET!!
De groeimonitor bestaat uit groeibalken voor 5 relevante taken binnen het beroep {beroep}. 
Varieer de groei ad random per taak (in %) en bereken het gemiddelde van deze taken en plaats dat gemiddelde in de variabele: {st.session_state.gemiddelde_groei}.
Gebruik markdown-formaat om een tabel weer te geven met de kolommen: Taak, Groei in % en Groeibalken (🪙).
Voorbeeld:

**GROEIMONITOR {beroep.upper()}**

| Taak   | Groei in % | 🪙        |
| :----- | :--------: | :-------: |
| Taak 1 | **23.5%**  | 🪙        |
| Taak 2 | **61.3%**  | 🪙🪙🪙    |
| Taak 3 | **48.5%**  | 🪙🪙     |
| Taak 4 | **87.1%**  | 🪙🪙🪙🪙  |
| Taak 5 | **17.7%**  | 🪙        |

Bewaar de groeimonitor in de variabele: {st.session_state.groei}. MAAR TOON DIE NIET AAN DE GEBRUIKER!!
Bewaar tevens het gemiddelde van de groei in: {trunc(st.session_state.gemiddelde_groei)}
"""
    messages = [{"role": "user", "content": prompt}]
    stream = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=cast(Any, messages),
        stream=True,
        temperature=0.6,
    )
    response = st.write_stream(stream)
    st.session_state.groei = str(response)[:-360]

    # Optimaliseer het extraheren van percentages met regex (inclusief decimalen)
    try:
        percentage_matches = re.findall(r"(\d+(?:\.\d+)?)%", str(response))
    except Exception as e:
        percentage_matches = []

    if percentage_matches:
        values = [float(g) for g in percentage_matches][:5]
        gemiddelde = sum(values) / len(values)
        if gemiddelde != st.session_state.gemiddelde_groei:

            st.session_state.old_gemiddelde_groei = st.session_state.gemiddelde_groei
            st.session_state.gemiddelde_groei = gemiddelde
            st.session_state.waarden = values


# ------------------------------------------------
# Weergave van studentinfo
# ------------------------------------------------
naam1, level1, level2 = st.columns([1, 1, 1])
with naam1:
    st.markdown("#### 🙋🏽**Student:**")
    st.markdown(f"###### :blue[{st.session_state.naam}]")
with level1:
    st.markdown("#### 🏫**Opleiding:**")
    st.markdown(f"###### **:blue[{st.session_state.opleiding}]**")
with level2:
    st.markdown("#### 🚗**Leerpad:**")
    st.markdown(f"###### **:blue[{str(st.session_state.level).replace(' ', '')}]**")

st.divider()

col_metric1, col_metric2 = st.columns([3, 2])
with col_metric1:
    groei_info = get_info_beroep(st.session_state.opleiding)
    if groei_info:
        st.markdown(groei_info)


with col_metric2:
    level = predict_learning_level(st.session_state.score, st.session_state.study_hours)

    st.metric(
        label="🏅**TOETS GEMIDDELDE in %**",
        value=f"🎯{st.session_state.score}% ({st.session_state.old_score})",
        delta=st.session_state.score - st.session_state.old_score,
        delta_color="normal",
        border=False,
    )
    st.metric(
        label="🏅**GROEI GEMIDDELDE in %**",
        value=f"🎯{str(st.session_state.gemiddelde_groei)[:4]}% ({str(st.session_state.old_gemiddelde_groei)[:4]})",
        delta=f"{str(st.session_state.gemiddelde_groei - st.session_state.old_gemiddelde_groei)[:4]}%",
        delta_color="normal",
        border=False,
    )
    st.metric(
        label="🏅**OEFENTOETS GROEI in %**",
        value=f"🎯{str(st.session_state.oefentoets_groei)}% ({str(st.session_state.old_oefentoets_groei)[:4]})",
        delta=f"{str(st.session_state.oefentoets_groei - st.session_state.old_oefentoets_groei)}%",
        delta_color="normal",
        border=False,
    )

st.divider()

# ------------------------------------------------
# AI-gegenereerd lesmateriaal en aanbevelingen
# ------------------------------------------------


@st.fragment
@logger
def personaliseer_lesmateriaal():
    st.header("🧙🏽‍♂️Gepersonaliseerd lesmateriaal")
    topic = upload_file()
    topic2 = st.text_input(
        ":blue[**Voer een onderwerp in en de AI genereert leesmateriaal op maat (bijv. thema veiligheid van burgerschap)**]",
        "",
    )

    if topic:
        st.session_state.topic = topic
        st.session_state.herschreven = generate_lesson_content(
            st.session_state.level, st.session_state.topic
        )
        return st.session_state.herschreven

    elif topic2:
        st.session_state.topic = f"{topic2}. \nDit alles binnen de context van de opleiding:\n{st.session_state.opleiding} op niveau {st.session_state.level}."
        st.session_state.herschreven = generate_lesson_content(
            st.session_state.level, st.session_state.topic
        )
        return st.session_state.herschreven

    if not st.session_state.topic and not st.session_state.herschreven:
        st.warning("Je moet een document uploaden of een onderwerp invoeren")
        st.markdown(
            "📚 **Als je geen bestand gebruikt kun je hier 👆🏽 een onderwerp invullen**"
        )

    @st.fragment
    @logger
    def doe_lesson_content(level, topic):
        if st.session_state.old_topic == topic and st.session_state.herschreven:
            return st.session_state.herschreven

        st.session_state.herschreven = generate_lesson_content(level, topic)

        return st.session_state.herschreven

    def handle_genereer_lesmateriaal():
        doe_lesson_content(st.session_state.level, st.session_state.topic)

    st.button(
        ":blue[**📌 GENEREER LESMATERIAAL**]", on_click=handle_genereer_lesmateriaal
    )

    # st.session_state.herschreven = personaliseer_lesmateriaal()
    if st.session_state.herschreven:
        with st.expander(
            label=f"✅ **AI-GEGENEREERD LESMATERIAAL OP NIVEAU: :blue[{st.session_state.level.upper()}]**",
            expanded=False,
        ):
            # st.markdown(st.session_state.herschreven)
            return st.session_state.herschreven


@logger
def doe_lesson_content(level, topic):
    if st.session_state.old_topic == topic and st.session_state.herschreven:
        return st.session_state.herschreven

    st.session_state.herschreven = generate_lesson_content(level, topic)
    return st.session_state.herschreven


st.session_state.herschreven = personaliseer_lesmateriaal()
st.markdown(
    "💡 **Als je geen bestand gebruikt kun je hier 👆🏽 een onderwerp invullen**"
)

if st.session_state.herschreven:
    with st.expander(
        label=f"✅ **AI-GEGENEREERD LESMATERIAAL OP NIVEAU: :blue[{str(st.session_state.level).upper()}]**",
        expanded=False,
    ):
        st.markdown(st.session_state.herschreven)


@st.fragment
@logger
def schoon_op():
    """
    Reset relevante variabelen in de session state.
    """
    st.session_state.herschreven = None
    st.session_state.topic = None
    st.session_state.eigen_file = None
    st.session_state.student_essay = None
    st.session_state.feedback = None
    st.session_state.feedback_response = None
    st.session_state.toets = None
    st.session_state.antwoorden_van_student = []
    st.session_state.correcte_antwoorden = []
    st.session_state.level = None
    return st.session_state


if st.button("🔄 :blue[**Vernieuw lesmateriaal**]", on_click=schoon_op):  # type: ignore
    pass


# ------------------------------------------------
# AI-Oefentoets met feedback
# ------------------------------------------------
st.header("📝 **AI-Oefentoets**")
st.write("**Test je kennis met een AI-oefentoets en ontvang AI-feedback!**")


@st.fragment
@logger
def genereer_oefentoets():
    if st.button(":blue[**📌 Genereer oefentoets**]"):
        
        if st.session_state.herschreven:
            maak_quiz = f"""Maak een oefentoets over:
            
{st.session_state.herschreven} 

op {st.session_state.level}-niveau. 

De toets bestaat uit 5 multiple-choice vragen waarbij elke vraag 4 antwoordmogelijkheden heeft (A, B, C of D).

1. Wat is de hoofdstad van Nederland?\n
A. Amsterdam    
B. Rotterdam    
C. Den Haag    
D. Utrecht    


Bewaar alle door jouw gemaakte toetsvragen in de lijst: {st.session_state.toets} 
en de juiste antwoorden in de lijst: {st.session_state.correcte_antwoorden}

(laat in eerste instantie de juiste antwoorden **NIET** zien aan de gebruiker, het mag pas nadat de gebruiker zelf alle vragen heeft beantwoord).

Gebruik markdown-formaat voor de opmaak.
"""
            quiz = client.chat.completions.create(
                model=MODEL, messages=[{"role": "system", "content": maak_quiz}]
            )
            st.session_state.toets = quiz.choices[0].message.content
            col_toets, col_antwoorden = st.columns([3, 1])
            with col_toets:
                st.markdown(st.session_state.toets[:-120])  # type: ignore
                st.markdown(
                    f"Toets gegenereerd in: {tijd_verschil_microsec(start_tijd)}"
                )

            @st.fragment
            @logger
            def antwoorden_invullen():
                st.markdown("")
                st.markdown("")
                for vraag_nr in range(1, 6):
                    if vraag_nr != 1:
                        st.markdown("")      
                    st.write(f"**Vraag {vraag_nr}**")
                    for optie in ["A", "B", "C", "D"]:
                        st.checkbox(
                            f"{vraag_nr}{optie}",
                            key=f"vraag{vraag_nr}{optie}",
                            on_change=lambda opt=f"{vraag_nr}{optie}": st.session_state.antwoorden_van_student.append(
                                opt
                            ),
                        )
                st.write(st.session_state.antwoorden_van_student)

            with col_antwoorden:
                topic = st.session_state.topic
                lengte = int(round(len(topic.splitlines()[0]))/10) + 3
                st.write(lengte)                
                # st.markdown("")
                st.markdown("### **Antwoorden:**")
                for i in range(0,lengte):
                    st.markdown("")
                antwoorden_invullen()
    # print(f"De TOETS:\n{st.session_state.toets}")
    # print(f"DE ANTWOORDEN STUDENT:\n{st.session_state.antwoorden_van_student}")
    # print(f"De JUISTE ANTWOORDEN:\n{st.session_state.correcte_antwoorden}")


genereer_oefentoets()


@st.fragment
@logger
def controleer_antwoorden():
    # st.markdown("📌 **Controleer de antwoorden:**")
    if st.button(":blue[**📌 Controleer antwoorden**]"):
        feedback_prompt = f"""Beoordeel de volgende antwoorden op deze toets en geef feedback aan de gebruiker op niveau {st.session_state.level}:
Toets:
{st.session_state.toets}
Antwoorden van de gebruiker: {st.session_state.antwoorden_van_student}
Juiste antwoorden: {st.session_state.correcte_antwoorden}

Geef feedback en verbeterpunten."""
        feedback_response = client.chat.completions.create(
            model=MODEL,
            messages=cast(Any, [{"role": "system", "content": feedback_prompt}]),
        )
        response = feedback_response.choices[0].message.content
        st.session_state.feedback_response = response
        st.markdown("📌 **AI-feedback:**")
        st.markdown(response)
        if (
            st.session_state.antwoorden_van_student
            == st.session_state.correcte_antwoorden
        ):
            # st.session_state.score += 10
            aantal_goed = len(
                members(
                    st.session_state.antwoorden_van_student,
                    st.session_state.correcte_antwoorden,
                )
            )
            if aantal_goed == 5:
                st.session_state.old_oefentoets_groei = (
                    st.session_state.oefentoets_groei
                )
                st.session_state.oefentoets_groei += 5
            if aantal_goed == 4:
                st.session_state.old_oefentoets_groei = (
                    st.session_state.oefentoets_groei
                )
                st.session_state.oefentoets_groei += 4
            if aantal_goed == 3:
                st.session_state.old_oefentoets_groei = (
                    st.session_state.oefentoets_groei
                )
                st.session_state.oefentoets_groei += 3
            if aantal_goed == 2:
                st.session_state.old_oefentoets_groei = (
                    st.session_state.oefentoets_groei
                )
                st.session_state.oefentoets_groei += 2
            if aantal_goed == 1:
                st.session_state.old_oefentoets_groei = (
                    st.session_state.oefentoets_groei
                )
                st.session_state.oefentoets_groei += 1

            st.session_state.feedback = f"🎉 Je hebt de toets goed gemaakt! Je score is nu {st.session_state.oefentoets_groei}%"
            st.markdown(st.session_state.feedback)
            st.rerun()
        return response


controleer_antwoorden()

# ------------------------------------------------
# Peer Learning Aanbevelingen
# ------------------------------------------------


@st.fragment
@logger
def peer_netwerk():
    st.header("👥 **Peer Learning Netwerk**")
    onderwerp = (
        st.session_state.topic if st.session_state.topic else st.session_state.opleiding
    )
    if st.button(
        f":blue[**Vind een klasgenoot voor extra oefening over:**] :orange[**{str(onderwerp).split('.')[0] if onderwerp else 'Geen onderwerp'}**]"
    ):
        students = ["Ali", "Betty", "Cees", "Danielle", "Edward", "Faouad"]
        student_match = np.random.choice(students)
        st.write(
            f"✨ Je kunt samenwerken met: **{student_match}** voor extra oefening over :orange[{str(onderwerp).split('.')[0]}]"
        )


peer_netwerk()

# ------------------------------------------------
# AI Feedback op ingeleverd werk
# ------------------------------------------------
st.header("🤝 Feedback op maat")
st.write(":blue[**De AI geeft feedback op jouw werk.**]")


@logger
@st.fragment
def geef_feedback():
    student_essay = st.session_state.student_essay
    if not student_essay:
        student_essay = upload_eigen_file()
        st.session_state.student_essay = student_essay
        if not student_essay:
            st.warning("Je moet een document uploaden")
            return

    if student_essay and st.session_state.feedback_eigen_werk:
        st.write("📌 **AI-feedback:**")
    elif student_essay:
        feedback_prompt = f"Geef feedback op het door de student ingeleverde werk: {student_essay}. Beoordeel inhoud, grammatica en structuur."
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": feedback_prompt}],
            temperature=0.2,
        )
        st.session_state.feedback_eigen_werk = response.choices[0].message.content
        st.markdown(f"Feedback gegenereerd in: {tijd_verschil_microsec(start_tijd)}")
    # The callback returns None as expected.


geef_feedback()

if st.session_state.student_essay and not st.session_state.feedback_eigen_werk:
    st.button(
        " :blue[**Geef feedback op mijn werk**]", on_click=geef_feedback
    )  # type: ignore
else:
    st.write("📌 **AI-feedback:**")
    st.markdown(st.session_state.feedback_eigen_werk)

if st.button("🔄 :blue[**Vernieuw feedback**]", on_click=schoon_op, key="vernieuw_feedback"):  # type: ignore
    pass

# ------------------------------------------------
# Gamificatie: Beloningssysteem
# ------------------------------------------------
st.header("🏆 **Gamificatie en Beloningen**")
progress = (st.session_state.score / 100) * 10
if progress >= 8.5:
    badge = "🏆🥇 Expert Badge!"
elif progress > 7.5:
    badge = "🥈 Gevorderde Badge!"
elif progress >= 6.5:
    badge = "🏅 Onderweg Badge!"
else:
    badge = "💡 Starter Badge!"
st.markdown(f"##### 🎖️ **Je hebt de volgende badge verdiend:** :orange[**{badge}**]")


@st.fragment
@logger
def make_grafiek_leaderbord(leaderboard_sorted):
    import matplotlib.pyplot as plt

    names = [name for name, _ in leaderboard_sorted]
    punten = [score for _, score in leaderboard_sorted]
    fig = plt.figure(
        figsize=(4, 3),
        layout="constrained",
        facecolor="white",
        edgecolor="grey",
        clear=True,
    )
    bar_colors = ["tab:green", "tab:orange", "tab:purple", "tab:orange", "tab:purple"]
    plt.bar(names, punten, align="center", color=bar_colors, edgecolor="black")
    plt.title("Punten per student")
    plt.xlabel("Studenten")
    plt.ylabel("Punten")
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 10))
    for i, v in enumerate(punten):
        plt.text(i, v + 2, str(v), ha="center", fontsize=8)
    st.pyplot(
        fig,
        use_container_width=True,
        clear_figure=True,
        dpi=100,
        format="png",
        bbox_inches="tight",
        pad_inches=0.00,
        facecolor="white",
        edgecolor="white",
    )


# Leaderboard (dummy data)
st.header("📈 **Leaderboard highest 5**")
punten = [
    89,
    51,
    st.session_state.score,
    72,
    43,
    19,
    35,
    63,
    81,
    91,
    23,
    11,
    45,
    55,
    35,
    78,
    99,
]
leaderboard = {
    "Ali": random.choice(punten),
    "Betty": random.choice(punten),
    "Jij": st.session_state.score,
    "Danielle": random.choice(punten),
    "Edward": random.choice(punten),
}
leaderboard_sorted = sorted(leaderboard.items(), key=lambda x: x[1], reverse=True)
st.session_state.leaderboard_sorted = leaderboard_sorted

leader1, leader2 = st.columns([2, 3])
with leader1:
    st.markdown("#### 🏆**Top studenten:**\n")
    if st.session_state.leaderboard_sorted:
        for name, points in leaderboard_sorted:
            st.markdown(f"**{name}:** :green[**{points} punten**]\n")
with leader2:
    make_grafiek_leaderbord(st.session_state.leaderboard_sorted)

st.divider()

# ------------------------------------------------
# Dagelijkse uitdaging
# ------------------------------------------------
dag1, dag2, dag3 = st.columns(3)
with dag1:
    st.subheader("🎯 **Dagelijkse Uitdaging**")
    daily_challenges = [
        f"Los 5 vragen over het beroep {st.session_state.opleiding} op!",
        f"Bekijk een uitlegvideo over het beroep {st.session_state.opleiding} en beantwoord een vraag.",
        f"Schrijf een korte samenvatting over een van de taken van het beroep {st.session_state.opleiding}.",
    ]
    daily_task = np.random.choice(daily_challenges)
    st.write(f"🔥 Vandaag’s uitdaging: {daily_task}")
with dag2:
    st.subheader("📅 **Niveaudoelen**")
    st.write("⚡ Voltooi 3 uitdagingen om naar het volgende niveau te gaan!")
with dag3:
    st.subheader("🥇 **Competitie Modus**")
    competition_mode = st.checkbox("Doe mee aan een wekelijkse AI-uitdaging!")
    if competition_mode:
        st.write(
            "🚀 Je doet nu mee aan de wekelijkse uitdaging! Voltooi zoveel mogelijk taken om te winnen!"
        )

# ------------------------------------------------
# AI Tutor Chatbot
# ------------------------------------------------

instructie_AI_coach = f"""
        Je bent een behulpzame en vriendelijke, Nederlands sprekende coach, je heet coach Karel van Oort, die een student helpt om na te denken over zijn leerproces. De bedoeling is dat de student zeer zelfsturend wordt, en regie neemt over zijn eigen leerproces. 
        Hieronder vind je de theorie achter een cyclisch model van zelfgestuurd leren.
        Dit model is een cyclus van vier stappen: oriënteren, plannen, monitoren en evalueren.
        
    <Theorie>
        ## DE ZELFSTURENDE STUDENT

        Cyclisch Model Zelfgestuurd Leren

        - Oriënteren (wat wil/  moet ik precies leren?).
        - Plannen van het leren: wat moet ik doen in welke volgorde?
        - Monitoren van het leren: ben ik op de goede weg?
        - Evalueren van het leren: is het mij gelukt? 


        ## Metacognitieve vaardigheden
        Zelfgestuurd leren als de mate waarin leerlingen metacognitief, motivationeel en gedragsmatig - dus in denken, voelen en handelen - betrokken zijn bij hun eigen leerproces. 

        ### Typen strategieën voor zelfgestuurd leren

        | Proces | 	Voorbeeld van gedrag 
        | :--   | :--
        | 1. Jezelf evalueren | Een leerling kijkt het eigen werk na
        | 2. Organiseren en transformeren |	Een leerling organiseert blokjes alvorens te gaan bouwen
        | 3. Doelen stellen en plannen | Een leerling stelt als doel om minstens drie sommen goed te        maken en houdt dat vervolgens bij
        | 4. Informatie zoeken | Een leerling wil iets te weten komen over een onderwerp en gaat daar actief en uit zichzelf naar op zoek
        | 5. Bijhouden en monitoren	| Een leerling checkt regelmatig of hij/zij de stof wel begrijpt
        | 6. Beïnvloeden van omgeving | Een leerling zoekt een rustig plekje om zich even goed op iets      te kunnen concentreren
        | 7. Jezelf regels opleggen	| Een leerling spreekt met zichzelf af dat hij/zij eerst iets af        moet hebben voordat hij/zij mag gaan spelen
        | 8. Herhalen en onthouden | Een leerling blijft een woord opschrijven om te kunnen onthouden hoe het geschreven moet worden
        | 9. Hulp vragen | Een leerling vraag iemand anders of die het wel snapt en wil helpen
        | 10. Terugkijken in materiaal	| Een leerling kijkt terug naar eerdere oefeningen
    </Theorie>
        
    
        Stel nu eerst jezelf voor. 
        Leg uit dat je hier bent als hun coach om hen te helpen zelfsturend te gaan werken.
        BELANGRIJK IS DAT JE HET GESPREK AANPAST AAN HET NIVEAU VAN DE STUDENT. DUS GEBRUIK GEEN MOEIELIJKE WOORDEN,  HET NIVEAU IS VAN EEN STUDENT MBO niveau {st.session_state.level} DIE OPLEIDING {st.session_state.opleiding} VOLGT. 
        
        Vraag dan de student om na te denken over een recente leer ervaring.
        Denk stap voor stap na en wacht op het antwoord van de student voordat je verder gaat met iets anders. 
        Deel je plan niet met de studenten. 
        Reflecteer op elke stap van het gesprek en beslis vervolgens wat je next wilt doen. Stel alleen 1 vraag tegelijk.

        Vraag de student om na te denken over de ervaring en om 1 uitdaging te noemen die ze heeft overwonnen en 1 uitdaging die ze niet heeft overwonnen. 
        Wacht op een reactie. 
        Ga niet verder voordat je een reactie krijgt, omdat je je volgende vraag moet aanpassen 
        op basis van het antwoord van de student.

        Vraag de student vervolgens om te reflecteren op deze ervaring en uitdaging. 
        Hoe is je begrip van jezelf veranderd? 
        Welke nieuwe inzichten heb je opgedaan? 
        Ga niet verder totdat je een reactie krijgt. 
        Deel je plan niet met de studenten. 
        Wacht altijd op een reactie, maar zeg de studenten niet dat je op een reactie wacht. 
        Stel open vragen, maar stel alleen één vraag tegelijk. 
        Moedig studenten aan om uitgebreide antwoorden te geven en belangrijke ideeën te articuleren. 
        Stel vervolgvragen. Bijvoorbeeld, als een student zegt dat ze een nieuw begrip hebben gekregen vraag dan of ze hun oude en nieuwe begrip kunnen uitleggen. 
        Vraag hen wat heeft geleid tot hun nieuwe inzicht. 
        Deze vragen stimuleren een diepere reflectie. Vraag om specifieke voorbeelden. 
        Bijvoorbeeld, als een student zegt dat hun kijk op hoe te plannen is veranderd, 
        vraag dan of ze een concreet voorbeeld uit hun ervaring kunnen geven 
        dat de verandering illustreert. 
        Specifieke voorbeelden verankeren reflecties in echte leermomenten.

        Bespreek obstakels. Vraag de student om na te denken over welke obstakels of twijfels ze nog steeds tegenkomt
        bij het toepassen van een vaardigheid. Bespreek strategieën om deze obstakels te overwinnen. 
        Dit helpt om reflecties om te zetten in het stellen van doelen. 
        Sluit het gesprek af door het reflectieve denken te prijzen. 
        Laat de student weten wanneer hun reflecties bijzonder doordacht zijn of vooruitgang tonen. 
        Laat de student weten of hun reflecties een verandering of groei in hun denken onthullen."""


@logger
def ai_chatbot(user_question):
    """
    AI Chatbot functie om vragen van studenten te beantwoorden.
    """

    if not st.session_state.messages:
        st.session_state.messages = [
            {"role": "system", "content": instructie_AI_coach},
        ]

    messages = st.session_state.messages
    messages.append(
        {"role": "user", "content": f"Beantwoord de vraag: {user_question}"}
    )
    response = client.chat.completions.create(model=MODEL, messages=cast(Any, messages))
    messages.append(
        {"role": "assistant", "content": response.choices[0].message.content or ""}
    )
    st.session_state.messages = messages
    print(st.session_state.messages)

    return response.choices[0].message.content


user_question = st.chat_input("Stel een vraag aan de AI Tutor")
if user_question:
    with st.chat_message("user"):
        st.write(user_question)
    chatbot_response = ai_chatbot(user_question)
    with st.chat_message("assistant"):
        st.write("🤖 AI Tutor antwoord:")
        st.write(chatbot_response)

with bottom():
    st.markdown(
        f"""
| :copyright: :grey[EdF 2025] | :grey[e.defeber@gmail.nl] | :grey[{vandaag_str}] |
| :------ | :------ | :------ |
"""
    )
