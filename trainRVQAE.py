import torch
import torch.nn as nn
from autoencoder import RVQAE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

loader = DataLoader(
    datasets.MNIST(r'C:\Users\yotam\Documents\University\self learning\Phoneme Hallucination optimization\codec gradient analysis',
                   download=False,
                   transform=transforms.ToTensor()),
    batch_size=64, shuffle=True
)

criterion = nn.MSELoss()
model = RVQAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10

os.makedirs('checkpoints', exist_ok=True)

print("Training:")
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (x, y) in enumerate(loader):
        reconstruction, codebook, rvq_loss, _ = model(x)
        loss = criterion(reconstruction, x) + rvq_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print(f"  Epoch {epoch+1} | Batch {i}/{len(loader)} | Loss {loss.item():.4f}")

    epoch_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1} complete | Avg Loss {epoch_loss:.4f}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, f'checkpoints/checkpoint_epoch_{epoch+1}_RVQ.pt')

print("Training done")