import requests
import numpy as np
import json
from tensorflow.keras.datasets import mnist

# URL corretto dell'InferenceGraph
url = "http://localhost:8080/v1/models/mnist-monitoring-graph:predict"

# Header con Host per il corretto routing nel grafo
headers = {
    "Content-Type": "application/json",
    "Host": "mnist-monitoring-graph.default.example.com"
}

def send_request(data):
    """
    Invia i dati al grafo e restituisce la risposta JSON.
    """
    payload = {"instances": data.tolist()}
    try:
        response = requests.post(url, headers=headers, json=payload)
        print("\nStatus code:", response.status_code)
        print("Raw response (prima 500 caratteri):", response.text[:500])
        return response.json()
    except requests.exceptions.JSONDecodeError as e:
        print("Errore parsing JSON:", e)
        return {"error": "Response empty or not JSON"}
    except Exception as e:
        print("Errore generico:", e)
        return {"error": str(e)}

# --- Test dati "normali" (dal dataset MNIST) ---
(_, _), (x_test, _) = mnist.load_data()
normal_sample = x_test[0:1].astype('float32') / 255.0
normal_sample = np.expand_dims(normal_sample, axis=-1)  # shape (1,28,28,1)

print("### Test con dati normali ###")
result_normal = send_request(normal_sample)
print(json.dumps(result_normal, indent=2))

# --- Test dati casuali per forzare drift ---
random_data = np.random.rand(1, 28, 28, 1).astype('float32')
print("### Test con dati casuali (drift) ###")
result_drift = send_request(random_data)
print(json.dumps(result_drift, indent=2))
