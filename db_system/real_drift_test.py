import torch
import torchvision.models as models
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- SETUP CNN ---
weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.classifier = torch.nn.Identity()
model.eval()

qdrant = QdrantClient("localhost", port=6333)
COLLECTION_NAME = "smart_drift_collection"

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1280, distance=Distance.COSINE),
    )

def get_embedding():
    # Simuliamo un'immagine con un po' di struttura invece di puro rumore
    img_tensor = torch.randn(1, 3, 224, 224) * 0.5 + 0.5 
    with torch.no_grad():
        return model(img_tensor).flatten().tolist()

# --- 1. FASE DI CALIBRAZIONE ---
print("Calibrazione del sistema sui dati 'normali'...")
similarities = []
# Prendiamo un vettore di riferimento fisso (il nostro "centro" del training set)
anchor_vector = get_embedding()

for _ in range(10):
    v = get_embedding()
    res = qdrant.query_points(collection_name=COLLECTION_NAME, query=v, limit=1).points
    # Se la collezione è vuota, usiamo l'anchor
    score = np.dot(v, anchor_vector) / (np.linalg.norm(v) * np.linalg.norm(anchor_vector))
    similarities.append(score)

mean_sim = np.mean(similarities)
std_sim = np.std(similarities)
threshold = mean_sim - (3 * std_sim) # Regola dei 3-sigma

print(f"Media Similarità: {mean_sim:.4f}")
print(f"Soglia Calcolata (Drift se < di): {threshold:.4f}\n")

# --- 2. TEST DI INFERENZA ---
def check_for_drift(v):
    # Calcolo manuale o via Qdrant
    sim = np.dot(v, anchor_vector) / (np.linalg.norm(v) * np.linalg.norm(anchor_vector))
    if sim < threshold:
        print(f"--- ⚠️ DRIFT RILEVATO! (Sim: {sim:.4f} < Threshold: {threshold:.4f})")
    else:
        print(f"--- ✅ DATO OK (Sim: {sim:.4f})")

# Proviamo a simulare un dato molto diverso (moltiplichiamo per un valore estremo)
print("Verifica nuovi dati:")
check_for_drift(get_embedding()) # Dovrebbe essere OK
check_for_drift(np.random.uniform(-1, 1, 1280).tolist()) # Questo DEVE essere Drift
