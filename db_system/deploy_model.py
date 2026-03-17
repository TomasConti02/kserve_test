import boto3
import os

# Configurazione MinIO
s3 = boto3.client('s3', 
                  endpoint_url='http://localhost:9000', 
                  aws_access_key_id='admin', 
                  aws_secret_access_key='password123')

BUCKET_NAME = 'models'
MODEL_FILE = 'mobilenet_v2_updated.pth'

def publish():
    # Crea il bucket 'models' se non esiste
    buckets = [b['Name'] for b in s3.list_buckets()['Buckets']]
    if BUCKET_NAME not in buckets:
        s3.create_bucket(Bucket=BUCKET_NAME)
        print(f"Bucket '{BUCKET_NAME}' creato.")

    # Carica il modello come 'latest'
    s3.upload_file(MODEL_FILE, BUCKET_NAME, "production/latest.pth")
    print(f"🚀 Modello '{MODEL_FILE}' caricato su MinIO come 'production/latest.pth'")

if __name__ == "__main__":
    publish()
