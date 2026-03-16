from fastapi import FastAPI, Request
from alibi_detect.cd import KSDrift
import joblib
import numpy as np

app = FastAPI()

# Carica il detector (deve essere presente nella stessa cartella del Dockerfile)
detector = joblib.load('my_drift_detector.pkl')

@app.post("/predict")
async def predict(request: Request):
    payload = await request.json()
    data = np.array(payload['instances'])
    
    # Esegue il rilevamento del drift
    drift_preds = detector.predict(data)
    
    # Ritorna il risultato al grafo di KServe
    return {
        "is_drift": bool(drift_preds['data']['is_drift']),
        "p_value": float(drift_preds['data']['p_value'][0])
    }
