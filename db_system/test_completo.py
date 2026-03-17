"""
lo script insegna a Qdrant com'è fatto un numero "normale" e poi gli chiede di fare la guardia contro dati assurdi o corrotti (il "Drift").


La funzione warm_up_qdrant è il cuore della configurazione.

    Cosa fa: Prende le prime 1000 immagini del dataset MNIST originale (quelle su cui il modello dovrebbe essere esperto).

    Il Vettore: Trasforma ogni immagine 28x28 (una griglia) in una linea piatta di 784 numeri (flatten).

    Indicizzazione: Carica questi 784 numeri su Qdrant.

    Perché: Qdrant ora ha una "mappa" dello spazio dei numeri validi. Se un dato cade in una zona della mappa dove non ci sono punti conosciuti, è sospetto.

. La Funzione check_drift (Il Controllo Qualità)

Questa funzione viene chiamata per ogni nuova immagine che arriva al sistema.

    La Ricerca: Prende l'immagine in arrivo e chiede a Qdrant: "Qual è l'immagine più simile che hai in memoria?".

    La Distanza Euclidea: Qdrant calcola quanto sono "lontani" i pixel dell'immagine nuova rispetto a quella più vicina in memoria.

        Distanza Bassa (es. 2.0 - 7.0): I pixel coincidono quasi del tutto con un numero reale. Tutto ok.

        Distanza Alta (es. > 12.0): L'immagine ha una disposizione di pixel che non assomiglia a nessun numero conosciuto.

3. Simulazione di Dati Reali vs Drift

Il loop finale mette alla prova il sistema alternando due tipi di "client":

    Dato REALE: Pesca un numero a caso dal test set. Qdrant trova quasi sempre un "vicino" molto simile, quindi la distanza è bassa.

    Dato DRIFT (np.random.rand): Genera "rumore bianco" (neve televisiva). Poiché il rumore non ha la forma di un numero, la distanza calcolata da Qdrant schizza verso l'alto.

Perché questo è fondamentale in AI?

In un sistema reale, il tuo modello KServe proverebbe a indovinare un numero anche se gli mandassi la foto di un gatto. Ti direbbe magari "è un 8" con una confidenza del 40%.

Grazie a questo script:

    Intercetti l'errore prima: Sai che quel dato è "diverso" (Drift) prima ancora di fidarti della risposta del modello.

    Monitoraggio costante: Puoi vedere se nel tempo i dati degli utenti stanno diventando diversi da quelli del training (es. utenti che iniziano a scrivere i numeri in corsivo o con tratti molto più spessi).


"""



import requests
import numpy as np
import uuid
import time
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from tensorflow.keras.datasets import mnist

# --- CONFIGURAZIONI ---
QDRANT_HOST = "localhost"
COLLECTION_NAME = "mnist_baseline"

# Inizializzazione client
qdrant = QdrantClient(QDRANT_HOST, port=6333)

def warm_up_qdrant(x_train):
    """Carica i primi 1000 esempi reali come baseline su Qdrant."""
    print(f"--- WARM-UP: Indicizzazione di 1000 immagini in '{COLLECTION_NAME}' ---")
    
    # Crea la collezione se non esiste
    if not qdrant.collection_exists(COLLECTION_NAME):
4

        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=784, distance=Distance.EUCLID)
        )
    else:
        print(f"La collezione '{COLLECTION_NAME}' esiste già. Salto creazione.")

    # Prepariamo i punti (vettori) da caricare
    points = []
    for i in range(1000):
        # Flattening da (28,28) a (784,)
        vector = x_train[i].flatten().tolist()
        points.append(PointStruct(id=i, vector=vector, payload={"type": "baseline"}))
    
    # Caricamento massivo (batch)
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print("Fine Warm-up. Ora Qdrant ha una memoria del mondo 'normale'.\n")

def check_drift(img_array):
    """Cerca il vicino più prossimo usando query_points (API più recente)."""
    vector = img_array.flatten().tolist()
    
    # Cerchiamo il punto più vicino nel database
    search_result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=1
    ).points

    if search_result:
        # Distance.EUCLID: 0 = identico, > 10-15 = probabile drift/rumore
        return search_result[0].score
    return 100.0

# --- MAIN ---
print("Caricamento dataset MNIST...")
(x_train, _), (x_test, _) = mnist.load_data()

# Normalizzazione (0.0 - 1.0)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 1. Esegui il Warm-up
try:
    warm_up_qdrant(x_train)
except Exception as e:
    print(f"Errore durante il warm-up: {e}")
    exit(1)

# 2. Test Loop
print("Inizio monitoraggio drift (CTRL+C per fermare)...")
print("-" * 50)

try:
    for i in range(1, 21):  # Eseguiamo 20 test
        # Alterna: una reale (indice casuale), una di rumore (drift)
        if i % 2 == 0:
            idx = np.random.randint(0, len(x_test))
            sample = x_test[idx]
            tipo = "REALE "
        else:
            # Genera rumore casuale uniforme
            sample = np.random.rand(28, 28).astype('float32')
            tipo = "DRIFT "

        distanza = check_drift(sample)
        
        # Logica di soglia: empiricamente ~12 per MNIST con distanza Euclidea
        status = "✅ OK" if distanza < 12.0 else "⚠️ DRIFT RILEVATO"
        
        print(f"Test {i:02d} | [{tipo}] Distanza: {distanza:6.2f} -> {status}")
        
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nMonitoraggio interrotto dall'utente.")
except Exception as e:
    print(f"Errore durante il loop: {e}")
