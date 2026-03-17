# Usa un'immagine Python ottimizzata
FROM python:3.11-slim

# Imposta la directory di lavoro
WORKDIR /app

# Installa le dipendenze di sistema necessarie per psycopg2 e imaging
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copia i requisiti e installali
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il codice della tua app
COPY app.py .

# Esponi la porta 8000
EXPOSE 8000

# Comando per avviare l'API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
