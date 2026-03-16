import requests
import numpy as np
from tensorflow.keras.datasets import mnist

# Carica un campione reale e normalizza
(_, _), (x_test, _) = mnist.load_data()
sample = x_test[0:1].reshape(1, -1).astype('float32') / 255.0

payload = {"instances": sample.tolist()}
resp = requests.post("http://localhost:8080/predict", json=payload)
print(resp.json())
