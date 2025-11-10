"""
Pipeline PyTorch Complète - Classification Fashion-MNIST
=========================================================
Ce script montre toutes les étapes d'un projet deep learning avec PyTorch :
1. Chargement des données
2. Définition du modèle
3. Entraînement
4. Évaluation
5. Visualisation des résultats
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Configuration du device (GPU si disponible, sinon CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Utilisation du device : {device}")

# ========================================
# 1. PRÉPARATION DES DONNÉES
# ========================================

# Transformations : conversion en tenseur et normalisation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalisation entre -1 et 1
])

# Téléchargement et chargement des datasets
print("\n📊 Chargement du dataset Fashion-MNIST...")
train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# DataLoaders pour charger les données par batch
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"   ✅ {len(train_dataset)} images d'entraînement")
print(f"   ✅ {len(test_dataset)} images de test")

# Noms des classes
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# ========================================
# VISUALISATION DES DONNÉES
# ========================================

print("\n🖼️  Visualisation d'exemples du dataset...")

# Créer une figure avec 2 lignes (train et test)
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# 3 images du set d'entraînement
for i in range(3):
    img, label = train_dataset[i]
    ax = axes[0, i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'TRAIN\n{class_names[label]}', fontsize=12, fontweight='bold')
    ax.axis('off')

# 3 images du set de test
for i in range(3):
    img, label = test_dataset[i]
    ax = axes[1, i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'TEST\n{class_names[label]}', fontsize=12, fontweight='bold')
    ax.axis('off')

plt.suptitle('Exemples du dataset Fashion-MNIST', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('dataset_examples.png', dpi=150, bbox_inches='tight')
print("   ✅ Exemples sauvegardés dans 'dataset_examples.png'")
plt.show()

# ========================================
# 2. DÉFINITION DU MODÈLE CNN
# ========================================

class FashionCNN(nn.Module):
    """
    Réseau de neurones convolutif pour classifier les images Fashion-MNIST
    Architecture : Conv -> Conv -> Flatten -> FC -> Dropout -> FC
    """
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        # Couches de convolution
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Couches de pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Couches fully connected
        # Après 2 pooling : 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Bloc 1 : Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))
        
        # Bloc 2 : Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))
        
        # Aplatir le tenseur
        x = x.view(-1, 64 * 7 * 7)
        
        # Couches fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Créer le modèle et le déplacer sur le device
model = FashionCNN().to(device)
print("\n🏗️  Architecture du modèle :")
print(model)
print(f"\n📊 Nombre de paramètres : {sum(p.numel() for p in model.parameters()):,}")

# ========================================
# 3. CONFIGURATION DE L'ENTRAÎNEMENT
# ========================================

# Fonction de perte
criterion = nn.CrossEntropyLoss()

# Optimiseur
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Pour stocker l'historique
train_losses = []
test_accuracies = []

# ========================================
# 4. FONCTIONS D'ENTRAÎNEMENT ET TEST
# ========================================

def train_epoch(model, device, train_loader, optimizer, criterion):
    """Entraîne le modèle pour une epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Déplacer les données sur le device
        data, target = data.to(device), target.to(device)
        
        # Réinitialiser les gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Mise à jour des poids
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def test(model, device, test_loader):
    """Évalue le modèle sur le dataset de test"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():  # Pas besoin de calculer les gradients
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# ========================================
# 5. ENTRAÎNEMENT DU MODÈLE
# ========================================

num_epochs = 10
print(f"\n🎯 Début de l'entraînement ({num_epochs} epochs)...")
print("=" * 60)

for epoch in range(1, num_epochs + 1):
    # Entraîner
    train_loss = train_epoch(model, device, train_loader, optimizer, criterion)
    train_losses.append(train_loss)
    
    # Tester
    test_acc = test(model, device, test_loader)
    test_accuracies.append(test_acc)
    
    print(f"Epoch {epoch}/{num_epochs} - Loss: {train_loss:.4f} - Test Acc: {test_acc:.2f}%")

print("=" * 60)
print(f"✅ Entraînement terminé ! Précision finale : {test_accuracies[-1]:.2f}%")

# ========================================
# 6. VISUALISATION DES RÉSULTATS
# ========================================

# Graphiques d'entraînement
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(range(1, num_epochs + 1), train_losses, 'b-', linewidth=2, marker='o')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Évolution de la perte d\'entraînement')
ax1.grid(True, alpha=0.3)

# Accuracy
ax2.plot(range(1, num_epochs + 1), test_accuracies, 'g-', linewidth=2, marker='o')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Évolution de la précision sur le test set')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("\n📈 Graphiques sauvegardés dans 'training_history.png'")

# ========================================
# 7. PRÉDICTIONS SUR DES EXEMPLES
# ========================================

# Récupérer quelques exemples
examples = iter(test_loader)
example_data, example_targets = next(examples)

# Faire des prédictions
model.eval()
with torch.no_grad():
    example_data = example_data.to(device)
    output = model(example_data)
    predictions = output.argmax(dim=1, keepdim=True)

# Visualiser 9 exemples
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    # Afficher l'image
    img = example_data[i].cpu().squeeze()
    ax.imshow(img, cmap='gray')
    
    # Titre avec prédiction et vraie classe
    true_label = class_names[example_targets[i]]
    pred_label = class_names[predictions[i].item()]
    color = 'green' if true_label == pred_label else 'red'
    
    ax.set_title(f'Vrai: {true_label}\nPrédit: {pred_label}', 
                 color=color, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
print("🔍 Exemples de prédictions sauvegardés dans 'predictions.png'")

# ========================================
# 8. SAUVEGARDER LE MODÈLE
# ========================================

torch.save(model.state_dict(), 'fashion_mnist_model.pth')
print("\n💾 Modèle sauvegardé dans 'fashion_mnist_model.pth'")

# Pour recharger le modèle plus tard :
# model = FashionCNN()
# model.load_state_dict(torch.load('fashion_mnist_model.pth'))
# model.eval()

print("\n🎉 Pipeline terminée avec succès !")
print("\n" + "=" * 60)
print("DIFFÉRENCES CLÉS PYTORCH vs TENSORFLOW :")
print("=" * 60)
print("1. Graphes dynamiques : Plus flexible, debug plus facile")
print("2. Syntaxe explicite : .zero_grad(), .backward(), .step()")
print("3. nn.Module : Définir forward() plutôt que call()")
print("4. DataLoader : Gestion élégante des batches")
print("5. Plus pythonique : Utilise des boucles Python natives")
print("=" * 60)