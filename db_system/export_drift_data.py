import psycopg2
import boto3
import os

# Connessioni
pg_conn = psycopg2.connect(host="localhost", database="drift_db", user="user", password="password123")
s3 = boto3.client('s3', endpoint_url='http://localhost:9000', aws_access_key_id='admin', aws_secret_access_key='password123')

cursor = pg_conn.cursor()
# 1. Recupera solo i nomi dei file che hanno subito drift
cursor.execute("SELECT image_name FROM drift_log WHERE status = 'needs_relabeling'")
rows = cursor.fetchall()

if not os.path.exists("./to_relabel"):
    os.makedirs("./to_relabel")

print(f"Recupero di {len(rows)} immagini per il relabeling...")

# 2. Scarica i file da MinIO
for row in rows:
    img_name = row[0]
    s3.download_file('drift-images', img_name, f"./to_relabel/{img_name}")
    print(f"Scaricato: {img_name}")

print("\nFatto! Le immagini sono nella cartella './to_relabel'")
