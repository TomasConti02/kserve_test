FROM python:3.10-slim

WORKDIR /app

# Copia i requisiti e installa
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia i file del codice e il modello
COPY app.py .
COPY my_drift_detector.pkl .

# Espone la porta e avvia l'app
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
