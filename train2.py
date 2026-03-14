import tensorflow as tf
import numpy as np
import os

# 1. Architettura bilanciata (Veloce su CPU, ma efficace)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    
    # Primo blocco: Estrazione feature
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.BatchNormalization(), # Aiuta la convergenza
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Secondo blocco: Più profondità
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    tf.keras.layers.Flatten(),
    
    # Regolarizzazione: evita l'overfitting
    tf.keras.layers.Dropout(0.3), 
    
    # Classificazione
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 2. Addestramento con dati reali (usiamo MNIST per vedere miglioramenti veri)
print("Caricamento dati MNIST per un test reale...")
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0

print("Addestramento in corso...")
model.fit(x_train, y_train, epochs=3, batch_size=64)

# 3. Salvataggio professionale
export_path = "./model_repo/2"
if not os.path.exists(export_path):
    os.makedirs(export_path)

model.export(export_path)

print(f"Modello avanzato salvato in: {export_path}")
