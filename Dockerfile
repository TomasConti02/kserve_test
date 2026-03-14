FROM alpine:latest
WORKDIR /app
# Copiamo il modello dentro l'immagine
#COPY model/ ./model/ prima versione
COPY model2/ ./model/ 
