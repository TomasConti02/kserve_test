import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os

# --- CONFIGURAZIONE ---
data_dir = './retrain_data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Caricamento del modello
# Usiamo i pesi corretti per evitare i warning di deprecazione
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
# Adattiamo l'ultimo strato (classifier) alle nostre 2 classi
model.classifier[1] = nn.Linear(model.last_channel, 2) 
model = model.to(device)

# 2. Trasformazioni
data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Caricamento dati con controllo errori
try:
    image_dataset = datasets.ImageFolder(data_dir, data_transforms)
    # batch_size=1 perché abbiamo pochissimi dati per il test
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=True)
    print(f"Dataset caricato: {len(image_dataset)} immagini trovate in {len(image_dataset.classes)} classi.")
except Exception as e:
    print(f"Errore nel caricamento del dataset: {e}")
    exit()

# 4. Fine-tuning
criterion = nn.CrossEntropyLoss()
# Alleniamo solo il classificatore per velocità (transfer learning)
optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

print("\nInizio Fine-tuning sui dati driftati...")
model.train()

for epoch in range(3):
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoca {epoch+1}/3 - Loss: {running_loss/len(dataloader):.4f}")

# 5. Salvataggio del modello aggiornato
torch.save(model.state_dict(), "mobilenet_v2_updated.pth")
print("\n✅ Modello aggiornato salvato come 'mobilenet_v2_updated.pth'")
