import requests
import numpy as np
import uuid
import time
import json
import boto3
import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from tensorflow.keras.datasets import mnist
import io

# --- CONFIGURAZIONI ---
KSERVE_URL = "http://localhost:8080/v1/models/mnist-monitoring-graph:predict"
HEADERS = {"Host": "mnist-monitoring-graph.default.example.com"}

# Client Inizializzazione
qdrant = QdrantClient("localhost", port=6333)
s3 = boto3.client('s3', endpoint_url='http://localhost:9000', 
                  aws_access_key_id='admin', aws_secret_access_key='password123')
pg_conn = psycopg2.connect(host="localhost", database="drift_db", user="user", password="password123")
pg_cur = pg_conn.cursor()

COLLECTION_NAME = "mnist_baseline"
BUCKET_NAME = "mnist-monitoring"

def setup_all():
    """Inizializza infrastruttura DB e Storage."""
    # Postgres
    pg_cur.execute("""
        CREATE TABLE IF NOT EXISTS inference_logs (
            id UUID PRIMARY KEY,
            prediction INT,
            drift_score FLOAT,
            is_drift BOOLEAN,
            image_url TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    pg_conn.commit()
    
    # MinIO
    if BUCKET_NAME not in [b['Name'] for b in s3.list_buckets()['Buckets']]:
        s3.create_bucket(Bucket=BUCKET_NAME)
    
    print("Infrastruttura pronta (Postgres & MinIO).")

def process_and_log(img_array, label_type):
    req_id = str(uuid.uuid4())
    
    # 1. DRIFT DETECTION (Qdrant)
    vector = img_array.flatten().tolist()
    search = qdrant.query_points(collection_name=COLLECTION_NAME, query=vector, limit=1).points
    drift_score = search[0].score if search else 100.0
    is_drift = drift_score > 12.0

    # 2. INFERENZA (KServe)
    # Prepariamo l'array per KServe (batch di 1 con canale: 1, 28, 28, 1)
    kserve_input = np.expand_dims(img_array, axis=(0, -1)).tolist()
    try:
        res = requests.post(KSERVE_URL, headers=HEADERS, json={"instances": kserve_input}, timeout=5)
        prediction = res.json().get("predictions", [None])[0]
        # Se KServe restituisce probabilità, prendiamo l'indice massimo
        if isinstance(prediction, list): prediction = np.argmax(prediction)
    except:
        prediction = -1 # Errore inferenza

    # 3. STORAGE (MinIO) - Salviamo come .npy in memoria
    mem_file = io.BytesIO()
    np.save(mem_file, img_array)
    mem_file.seek(0)
    s3_path = f"requests/{req_id}.npy"
    s3.upload_fileobj(mem_file, BUCKET_NAME, s3_path)

    # 4. PERSISTENZA (Postgres)
    pg_cur.execute(
        "INSERT INTO inference_logs (id, prediction, drift_score, is_drift, image_url) VALUES (%s, %s, %s, %s, %s)",
        (req_id, int(prediction), float(drift_score), is_drift, s3_path)
    )
    pg_conn.commit()

    return req_id, prediction, drift_score, is_drift

# --- MAIN ---
setup_all()
(_, _), (x_test, _) = mnist.load_data()
x_test = x_test.astype('float32') / 255.0

print("\nAvvio monitoraggio integrato...")
try:
    for i in range(1, 11):
        is_real = (i % 2 == 0)
        sample = x_test[np.random.randint(0, 10000)] if is_real else np.random.rand(28, 28).astype('float32')
        
        rid, pred, score, drift = process_and_log(sample, "REAL" if is_real else "DRIFT")
        
        icon = "⚠️" if drift else "✅"
        print(f"[{i:02d}] Pred: {pred} | Drift: {score:5.2f} | ID: {rid[:8]}... {icon}")
        time.sleep(1)

except KeyboardInterrupt:
    print("\nFermo.")
finally:
    pg_cur.close()
    pg_conn.close()
