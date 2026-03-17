import requests
import numpy as np
import json
import time
import random
from tensorflow.keras.datasets import mnist

# Configurazioni
URL = "http://localhost:8080/v1/models/mnist-monitoring-graph:predict"
HEADERS = {
    "Content-Type": "application/json",
    "Host": "mnist-monitoring-graph.default.example.com"
}
NUM_REQUESTS = 50           # Numero totale di richieste (0 = infinite)
DRIFT_PROBABILITY = 0.3     # Probabilità di inviare un campione casuale (drift)
SLEEP_BETWEEN = 1.0         # Secondi tra una richiesta e l'altra

# Carica MNIST una volta
(_, _), (x_test, _) = mnist.load_data()
x_test = x_test.astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1)  # shape (N, 28, 28, 1)

def create_normal_sample():
    """Preleva un campione casuale dal test set MNIST."""
    idx = random.randint(0, len(x_test) - 1)
    return x_test[idx:idx+1]

def create_drift_sample():
    """Genera un campione casuale (rumore) per simulare drift."""
    return np.random.rand(1, 28, 28, 1).astype('float32')

def send_request(data):
    """
    Invia i dati al grafo e restituisce la risposta JSON.
    """
    payload = {"instances": data.tolist()}
    try:
        response = requests.post(URL, headers=HEADERS, json=payload, timeout=5)
        print(f"Status: {response.status_code}")
        # Mostra solo un breve estratto della risposta per non intasare il terminale
        resp_text = response.text[:200] + "..." if len(response.text) > 200 else response.text
        print(f"Response: {resp_text}")
        return response.json() if response.text else {}
    except requests.exceptions.JSONDecodeError as e:
        print("Errore parsing JSON:", e)
        return {"error": "Response empty or not JSON"}
    except Exception as e:
        print("Errore generico:", e)
        return {"error": str(e)}

def main():
    print(f"Avvio invio richieste in loop. Numero richieste: {NUM_REQUESTS if NUM_REQUESTS>0 else 'illimitato'}")
    print(f"Probabilità drift: {DRIFT_PROBABILITY}, pausa: {SLEEP_BETWEEN}s")

    counter = 0
    drift_count = 0
    normal_count = 0

    while True:
        # Decidi se inviare dato normale o drift
        if random.random() < DRIFT_PROBABILITY:
            sample = create_drift_sample()
            sample_type = "DRIFT"
            drift_count += 1
        else:
            sample = create_normal_sample()
            sample_type = "NORMALE"
            normal_count += 1

        print(f"\n[{counter+1}] Invio campione {sample_type}...")
        result = send_request(sample)
        # Se vuoi stampare l'intera risposta, decommenta la riga sotto:
        # print(json.dumps(result, indent=2))

        counter += 1
        if NUM_REQUESTS > 0 and counter >= NUM_REQUESTS:
            break

        time.sleep(SLEEP_BETWEEN)

    print("\n=== Riepilogo ===")
    print(f"Totale richieste: {counter}")
    print(f"  Normali: {normal_count}")
    print(f"  Drift:   {drift_count}")

if __name__ == "__main__":
    main()
