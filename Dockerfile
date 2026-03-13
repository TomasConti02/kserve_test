FROM alpine:latest
WORKDIR /app
# Copiamo il modello dentro l'immagine
COPY model/ ./model/
