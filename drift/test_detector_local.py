import requests
import numpy as np
from tensorflow.keras.datasets import mnist

# Carica dati di test MNIST e normalizza
(_, _), (x_test, _) = mnist.load_data()
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Preleva un campione
sample = x_test[0:1]  # primo esempio

# Prepara il payload
payload = {"instances": sample.tolist()}

# Invia richiesta al detector locale
url = "http://localhost:8080/predict"
response = requests.post(url, json=payload)
print("Risposta del detector:")
print(response.json())

# Verifica che non segnali drift (dovrebbe essere False con p-value alto)
