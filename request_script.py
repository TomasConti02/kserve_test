import requests
import numpy as np
from PIL import Image
import json

def predict_image(image_path, url):
    # 1. Carica e pre-processa (trasforma in scala di grigi, 28x28, normalizza)
    img = Image.open(image_path).convert('L').resize((28, 28))
    img_data = np.array(img).astype('float32') / 255.0
    img_data = img_data.reshape(1, 28, 28, 1)

    # 2. Crea il payload JSON
    payload = {"instances": img_data.tolist()}

    # 3. Invia la richiesta
    headers = {"Host": "simple-cnn.default.example.com"}
    response = requests.post(url, json=payload, headers=headers)
    
    # 4. Analisi
    predictions = response.json()['predictions'][0]
    predicted_class = np.argmax(predictions)
    print(f"Predizione: {predicted_class} (Confidenza: {predictions[predicted_class]:.2%})")

# Esecuzione
predict_image("test.png", "http://localhost:8080/v1/models/simple-cnn:predict")
