import requests
import numpy as np

# Punti a i due servizi singolarmente
URL_PREDICTOR = "http://localhost:8080/v1/models/simple-cnn:predict"
URL_DRIFT = "http://localhost:8080/v1/models/drift-detector:predict"

def test_monitoring_manually(data):
    payload = {"instances": data.tolist()}
    
    # 1. Chiamata al modello
    resp_pred = requests.post(URL_PREDICTOR, json=payload, headers={"Host": "simple-cnn.default.example.com"})
    
    # 2. Chiamata al detector
    resp_drift = requests.post(URL_DRIFT, json=payload, headers={"Host": "drift-detector.default.example.com"})
    
    return resp_pred.json(), resp_drift.json()

# Esempio di utilizzo
data_test = np.zeros((1, 28, 28, 1), dtype="float32")
pred, drift = test_monitoring_manually(data_test)
print(f"Predizione: {pred}")
print(f"Drift: {drift}")
