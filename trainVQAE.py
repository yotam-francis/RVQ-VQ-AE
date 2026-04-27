import torch
import torch.nn as nn
import torch.nn.functional as F
from autoencoder import VQAE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

loader = DataLoader(
datasets.MNIST(r'C:\Users\yotam\Documents\University\self learning\Phoneme Hallucination optimization\codec gradient analysis',
               download=False, 
               transform=transforms.ToTensor()),batch_size=64,shuffle=True) # Load MNIST #TODO:replace with placeholder for github
criterion = nn.MSELoss()
model = VQAE()
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4)
num_epochs = 10
print("Training:")
for epoch in range(num_epochs):
    running_loss = 0.0
    for x,y in loader:

        reconstruction,codebook,vq_loss = model(x)
        L2Loss = criterion(reconstruction,x)
        loss = vq_loss + L2Loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(loader)
    print(f"Epochs {epoch+1} | Loss {(epoch_loss):.4f}")
   
    # save weights after every epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, f'checkpoint_epoch_{epoch+1}.pt')
    print(f"Saved checkpoint_epoch_{epoch+1}.pt")
    
print("Training done")


    

