# Gebruik een officiële Python runtime als basis-image
FROM python:3.11-slim

# Installeer systeembibliotheken die nodig zijn voor o.a. audio en beeldverwerking
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Stel de werkdirectory in de container in
WORKDIR /app

# Kopieer het requirements-bestand en installeer de Python-packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopieer de rest van de applicatiecode naar de container
COPY . .

# Stel poort 8501 beschikbaar voor de buitenwereld
EXPOSE 8501

# Start de Streamlit-app wanneer de container wordt gestart
# De app wordt toegankelijk vanaf buiten de container
CMD ["streamlit", "run", "app/duo.py", "--server.address=0.0.0.0", "--server.port=8501"]
