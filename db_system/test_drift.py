import boto3
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import psycopg2
import numpy as np

# --- CONFIGURAZIONE ---
s3 = boto3.client('s3', endpoint_url='http://localhost:9000', 
                  aws_access_key_id='admin', aws_secret_access_key='password123')

qdrant = QdrantClient("localhost", port=6333)

pg_conn = psycopg2.connect(host="localhost", database="drift_db", user="user", password="password123")
cursor = pg_conn.cursor()

COLLECTION_NAME = "cnn_embeddings"

def init_systems():
    # 1. MinIO
    if 'drift-images' not in [b['Name'] for b in s3.list_buckets()['Buckets']]:
        s3.create_bucket(Bucket='drift-images')
    
    # 2. Postgres
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS drift_log (
            id SERIAL PRIMARY KEY,
            image_name TEXT,
            drift_score FLOAT,
            status TEXT
        )
    """)
    pg_conn.commit()

    # 3. Qdrant (Nuovo metodo senza deprecation)
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )

def simulate_inference(img_name, is_drifted=False):
    # Vettore "normale" vs "drifted"
    vector = [0.1, 0.1, 0.1, 0.1] if not is_drifted else [0.9, 0.8, 0.7, 0.9]
    
    # A. Salva su MinIO
    s3.put_object(Bucket='drift-images', Key=img_name, Body=b"binary_data_of_image")
    
    # B. Carichiamo un punto di riferimento se la collezione è vuota (per il calcolo distanza)
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(id=1, vector=[0.1, 0.1, 0.1, 0.1], payload={"type": "training_sample"})]
    )
    
    # C. Ricerca similarità (Metodo aggiornato)
    search_result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=1
    ).points
    
    score = search_result[0].score if search_result else 0
    print(f"Analisi {img_name} - Score di similarità (Cosine): {score:.4f}")

    # D. Log Drift se lo score è basso
    if score < 0.90:
        print(f"⚠️ DRIFT RILEVATO per {img_name}!")
        cursor.execute(
            "INSERT INTO drift_log (image_name, drift_score, status) VALUES (%s, %s, %s)",
            (img_name, score, "needs_relabeling")
        )
        pg_conn.commit()

# --- ESECUZIONE ---
init_systems()
print("Sistemi inizializzati correttamente.\n")

simulate_inference("normal_car.jpg", is_drifted=False)
simulate_inference("weird_shadow_car.jpg", is_drifted=True)

cursor.execute("SELECT * FROM drift_log")
print("\nLog Drift recuperati da Postgres:", cursor.fetchall())
