from fastapi import FastAPI, Request
from alibi_detect.cd import KSDrift
import joblib
import numpy as np

app = FastAPI()

# Carica il detector
detector = joblib.load('my_drift_detector.pkl')

@app.post("/predict")
async def predict(request: Request):
    payload = await request.json()
    data = np.array(payload['instances'])
    
    if data.shape[1] != 784:
        data = data.reshape(data.shape[0], 784)
    
    drift_preds = detector.predict(data)
    
    # Log di debug (opzionale, puoi rimuoverli in produzione)
    print(f"DEBUG: Struttura di drift_preds: {drift_preds.keys()}")
    print(f"DEBUG: Contenuto di 'data': {drift_preds['data'].keys()}")
    
    # Estrai i valori corretti (sono array, prendiamo il primo elemento)
    is_drift = bool(drift_preds['data']['is_drift'][0])
    p_value = float(drift_preds['data']['p_val'][0])
    
    return {
        "is_drift": is_drift,
        "p_value": p_value
    }
