from fastapi import FastAPI, Request
from alibi_detect.cd import KSDrift
import joblib
import numpy as np

app = FastAPI()
detector = joblib.load('my_drift_detector.pkl')

@app.post("/v1/models/drift-detector:predict")
async def predict_kserve(request: Request):
    return await predict(request)


@app.post("/predict")
async def predict(request: Request):
    payload = await request.json()
    data = np.array(payload['instances']).astype('float32')
    
    # Assicura che i dati siano in un batch 2D (n_samples, n_features)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # Se l'ultima dimensione non è 784, tenta di appiattire
    if data.shape[-1] != 784:
        if data.ndim == 3 and data.shape[1:] == (28, 28):
            data = data.reshape(data.shape[0], -1)
        else:
            data = data.reshape(data.shape[0], 784)
    
    # NORMALIZZAZIONE: divide per 255.0 come nel training
    data = data / 255.0
    
    drift_preds = detector.predict(data)
    
    # Gestione flessibile di is_drift
    is_drift_val = drift_preds['data']['is_drift']
    if isinstance(is_drift_val, (list, np.ndarray)):
        is_drift = bool(is_drift_val[0])
    else:
        is_drift = bool(is_drift_val)
    
    # Gestione flessibile di p_val
    p_val_val = drift_preds['data']['p_val']
    if isinstance(p_val_val, (list, np.ndarray)):
        p_value = float(p_val_val[0])
    else:
        p_value = float(p_val_val)
    
    return {"is_drift": is_drift, "p_value": p_value}
