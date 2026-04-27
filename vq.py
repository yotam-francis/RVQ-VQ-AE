import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_weight=0.25):
        super().__init__()
        # codebook: num_embeddings vectors each of size embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        
        # initialize codebook with uniform distribution
        nn.init.uniform_(self.embedding.weight, -1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z):
        # z shape: (batch, embedding_dim, time)
        z_flat = z.permute(0,2,1) # (batch, time, embedding_dim)
        b,t,d = z_flat.shape 
        z_flat =  z_flat.reshape(-1,d) # (batch*time, embedding_dim)
        z_flat = z_flat.unsqueeze(1) # (batch*time,1, embedding_dim)
        weights = self.embedding.weight.unsqueeze(0) # (1, num_embeddings(codes), embedding_dim)
        distances = (z_flat - weights) # Broadcasting - (batch*time,num_embedding, embedding_dim)
        distances = torch.sum(torch.pow(distances,2),dim = 2) # (batch*time, num_embedding) sum over the latent space representaion distances

        # Argmin is non-diffrentiable there for we will use STE to pass the gradient backwards
        k = distances.argmin(dim = 1) # (batch*time, 1) return the minimal distance embedding indices in the latent space for each point in time
        z_q = self.embedding.weight[k] # (batch*time, embedding_dim) - nearest codebook vector for each frame
        z_flat_q = z_flat.squeeze(1) # Squeeze to match z_q for broadcasting (batch*time,embdedding_dim)
        z_q_st = z_flat_q + (z_q - z_flat_q).detach() # Straight trough estimator (STE) forward pass is z_q backwards grad is ~z_flat (batch*time, embedding_dim)

        # Compute losses
        # Commitment loss
        commitment_loss = torch.mean((z_flat_q - z_q.detach())**2) # Freeze codebook gradients to measure loss to commitment to codebook (z_q)
        # Codebook loss
        codebookd_loss = torch.mean((z_flat_q.detach() - z_q)**2) # Freeze feature gradients to measure loss of codebook to gradients (z or z_flat_q while matching dims)
        # Loss
        loss = codebookd_loss + self.commitment_weight*commitment_loss # Total loss with commitment loss weighted

        # Reshape STE and codebook indices
        z_q_st = z_q_st.reshape(b,t,d).permute(0,2,1)
        k = k.reshape(b,t)

        return z_q_st,k,loss # Return feature vector (for training), Codebook entries (for trasmmission), loss
