import tensorflow as tf
import numpy as np

# 1. Definisci una CNN (aggiornato 'input_shape' a 'shape' per Keras 3)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 2. Addestramento fittizio
print("Addestramento fittizio in corso...")
x_train = np.random.random((100, 28, 28, 1))
y_train = np.random.randint(10, size=(100,))
model.fit(x_train, y_train, epochs=1)

# 3. Salva il modello per KServe (Usa model.export per Keras 3)
export_path = "./model_repo/1"

# QUESTA E' LA RIGA CAMBIATA
model.export(export_path) 

print(f"Modello pronto e salvato in: {export_path}")
