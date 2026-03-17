from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import boto3
from qdrant_client import QdrantClient
import psycopg2
import numpy as np

app = FastAPI()

# --- SETUP (In produzione usa variabili d'ambiente) ---
THRESHOLD = 0.9783  # La soglia che abbiamo calcolato prima
s3 = boto3.client('s3', endpoint_url='http://localhost:9000', aws_access_key_id='admin', aws_secret_access_key='password123')
qdrant = QdrantClient("localhost", port=6333)
pg_conn = psycopg2.connect(host="localhost", database="drift_db", user="user", password="password123")

# Carichiamo il modello una sola volta all'avvio
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier = torch.nn.Identity()
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Leggi l'immagine
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # 2. Estrai Embedding
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(input_tensor).flatten().tolist()
    
    # 3. Controllo Drift (confronto con anchor o Qdrant)
    # Qui semplifichiamo cercando la similarità massima nel DB
    search_res = qdrant.query_points(collection_name="smart_drift_collection", query=embedding, limit=1).points
    score = search_res[0].score if search_res else 0.0

    status = "OK"
    if score < THRESHOLD:
        status = "DRIFT_DETECTED"
        # 4. Azione Automatica: Salva su MinIO e Logga su Postgres
        s3.put_object(Bucket='drift-images', Key=file.filename, Body=image_data)
        
        cur = pg_conn.cursor()
        cur.execute("INSERT INTO drift_log (image_name, drift_score, status) VALUES (%s, %s, %s)",
                    (file.filename, score, "needs_relabeling"))
        pg_conn.commit()
        cur.close()

    return {
        "filename": file.filename,
        "similarity_score": round(score, 4),
        "status": status,
        "action": "Saved to MinIO for retrain" if status == "DRIFT_DETECTED" else "None"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
