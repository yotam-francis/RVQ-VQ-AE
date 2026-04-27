import torch
import torch.nn as nn
import torch.nn.functional as F
from vq import VectorQuantizer
from rvq import RVQ
class Enccoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,stride=2,kernel_size=3,padding=1) 
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,stride=2,kernel_size=3,padding=1) 
    def forward(self,x):
        x = F.relu(self.conv1(x)) # (B,1,28,28) -> (B,32,14,14)
        x = F.relu(self.conv2(x)) # (B,32,14,14) -> (B,64,7,7)
        x = x.flatten(start_dim = 2) # (B,64,7,7) -> (B,64,49)
        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels=64,out_channels=32,stride=2,kernel_size=3,padding=1,output_padding=1) # (B,1,28,28) -> (B,32,14,14)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=1,stride=2,kernel_size=3,padding=1,output_padding=1) # (B,1,28,28) -> (B,32,14,14)
    def forward(self,x):
        x = x.reshape(-1,64,7,7) # (B,64,49) -> (B,64,7,7)
        x = F.relu(self.deconv1(x)) # (B,64,7,7) -> (B,32,14,14)
        x = torch.sigmoid(self.deconv2(x)) # (B,32,14,14) -> (B,1,28,28)
        return x
    
class VQAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Enccoder() # (B,1,28,28) -> (B,64,49)
        self.vq = VectorQuantizer(embedding_dim=64,num_embeddings=512)
        self.decoder = Decoder() # (B,64,49) -> (B,1,28,28)
    def forward(self,x):
        x = self.encoder(x)
        x,k,loss = self.vq(x)
        x = self.decoder(x)
        return x,k,loss
    
class RVQAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Enccoder() # (B,1,28,28) -> (B,64,49)
        self.rvq = RVQ(num_quantizers=4,num_embed_start=256)
        self.decoder = Decoder() # (B,64,49) -> (B,1,28,28)
    def forward(self, x):
        z = self.encoder(x)
        z_q, k, loss, intermediates = self.rvq(z)
        reconstruction = self.decoder(z_q)
        
        # decode each intermediate to visualize per-stage quality
        intermediate_recons = [self.decoder(inter) for inter in intermediates]
        
        return reconstruction, k, loss, intermediate_recons