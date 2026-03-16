import numpy as np
from alibi_detect.cd import KSDrift
import joblib
import tensorflow as tf

# Carica MNIST come esempio (o usa i tuoi dati di training)
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
# Normalizzazione e reshape: importante che sia identico a come prepari i dati per il modello CNN
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0

# Inizializza il detector
cd = KSDrift(x_train, p_val=.05)

# Salva il file
joblib.dump(cd, 'my_drift_detector.pkl')
print("File 'my_drift_detector.pkl' creato con successo.")
