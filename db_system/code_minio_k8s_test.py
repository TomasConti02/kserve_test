import io
import os
from minio import Minio
from PIL import Image, ImageDraw

# --- Configurazione MinIO ---
endpoint = os.getenv('MINIO_ENDPOINT', 'minio.drift-system.svc.cluster.local:9000')
access_key = os.getenv('MINIO_ACCESS_KEY')
secret_key = os.getenv('MINIO_SECRET_KEY')

client = Minio(
    endpoint,
    access_key=access_key,
    secret_key=secret_key,
    secure=False
)

# --- Crea un'immagine di prova (100x100, cerchio rosso su sfondo blu) ---
img = Image.new('RGB', (100, 100), color='blue')
draw = ImageDraw.Draw(img)
draw.ellipse((25, 25, 75, 75), fill='red', outline='white')

# Salva l'immagine in un buffer BytesIO
buffer = io.BytesIO()
img.save(buffer, format='PNG')
image_data = buffer.getvalue()
buffer.close()

print(f"Immagine creata: {len(image_data)} bytes")

# --- Carica su MinIO ---
bucket_name = "test-images"
if not client.bucket_exists(bucket_name):
    client.make_bucket(bucket_name)
    print(f"Bucket '{bucket_name}' creato.")

# Usa un nome univoco
object_name = "test_image.png"

# Carica l'immagine (deve essere un file-like, usiamo BytesIO)
data_stream = io.BytesIO(image_data)
client.put_object(
    bucket_name,
    object_name,
    data=data_stream,
    length=len(image_data),
    content_type="image/png"
)
print(f"Immagine caricata come '{object_name}' nel bucket '{bucket_name}'.")

# --- Recupera l'immagine da MinIO ---
obj = client.get_object(bucket_name, object_name)
downloaded_data = obj.read()
obj.close()

print(f"Immagine scaricata: {len(downloaded_data)} bytes")

# --- Confronta i dati originali e scaricati ---
if image_data == downloaded_data:
    print("✅ Verifica superata: i dati originali e scaricati sono identici.")
else:
    print("❌ ERRORE: i dati differiscono!")

# (Opzionale) Salva l'immagine scaricata su un file locale per ispezione visiva
with open("/tmp/test_image_downloaded.png", "wb") as f:
    f.write(downloaded_data)
print("Immagine salvata anche in /tmp/test_image_downloaded.png (puoi copiarla dal pod con kubectl cp)")
