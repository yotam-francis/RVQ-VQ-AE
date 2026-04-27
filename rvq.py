import torch
import torch.nn as nn
from vq import VectorQuantizer

class RVQ(nn.Module):
    def __init__(self, num_embed_start, num_quantizers=6):
        super().__init__()
        self.num_embed_start = num_embed_start
        sizes = [max(1, num_embed_start // (2**i)) for i in range(num_quantizers)]
        self.quantizers = nn.ModuleList([
            VectorQuantizer(sizes[i], 64) for i in range(num_quantizers)
        ]) # Coarse to fine decrease codebook size as we go

    def forward(self, x):
        residual = x
        z_q_sum = torch.zeros_like(x)
        codes = []
        total_loss = 0
        intermediates = []  # cumulative reconstruction after each stage

        for vq in self.quantizers:
            z_q, k, loss = vq(residual)
            z_q_sum += z_q
            residual = x - z_q_sum
            codes.append(k)
            total_loss += loss
            intermediates.append(z_q_sum.clone())  # snapshot of reconstruction so far

        return z_q_sum, codes, total_loss, intermediates