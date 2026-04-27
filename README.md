# RVQ-VQ-AE

Implementation of Vector Quantization (VQ) and Residual Vector Quantization (RVQ) autoencoders from scratch in PyTorch, trained on MNIST. Built as a learning project to understand how VQ and RVQ work mechanistically -- from the argmin lookup and straight-through estimator to codebook collapse and residual quantization -- as a foundation for working with discrete representation learning in neural audio codecs.

## What this is

Neural audio codecs like EnCodec and WavTokenizer compress speech into sequences of discrete tokens using VQ at the bottleneck. This project implements the core quantization machinery from first principles and demonstrates the key failure mode (codebook collapse) and the structural solution (residual quantization).

## Key findings

- Single VQ trained on MNIST exhibits severe codebook collapse: only 13/512 entries (2.5%) actively used after 10 epochs
- RVQ with 4 stages achieves 87.4% lower MSE than single VQ with the same parameter count (68K vs 70K)
- RVQ utilization pattern reveals why residual quantization works: Stage 1 collapses (1.6% utilization) while later stages achieve 69-84% utilization because residuals have lower variance and more uniform distribution
- Shrinking the codebook hides collapse but does not fix it -- the encoder converges to the same active representations regardless of nominal codebook size

## Architecture

```
Input (B, 1, 28, 28)
    -> Encoder: Conv2d x2 with stride=2
    -> Latent (B, 64, 49)  -- 49 spatial positions, each 64-dim
    -> VQ or RVQ bottleneck
    -> Decoder: ConvTranspose2d x2
    -> Reconstruction (B, 1, 28, 28)
```

**VQ bottleneck:** Each of the 49 spatial positions is independently quantized to the nearest entry in a codebook of 512 x 64-dim vectors. Gradient flows through the non-differentiable argmin via the straight-through estimator (STE).

**RVQ bottleneck:** 4 VQ stages in cascade. Each stage quantizes the residual error of the previous stage. Codebook sizes decrease across stages (256, 128, 64, 32) following the coarse-to-fine principle.

## Files

```
vq.py           -- VectorQuantizer: argmin lookup, STE, commitment and codebook losses
rvq.py          -- ResidualVQ: cascaded VQ stages with decreasing codebook sizes
autoencoder.py  -- VQAE and RVQAE: encoder/decoder wrapping the quantizers
trainVQAE.py    -- Training loop for single VQ model
trainRVQAE.py   -- Training loop for RVQ model
results.ipynb   -- Reconstruction visualization and codebook utilization analysis
```

## Usage

```bash
pip install torch torchvision matplotlib
```

Train VQ model:
```bash
python trainVQAE.py
```

Train RVQ model:
```bash
python trainRVQAE.py
```

View results (requires trained checkpoints):
```bash
jupyter notebook results.ipynb
```

## Concepts demonstrated

**Straight-through estimator:** The argmin operation has zero gradient almost everywhere. STE passes gradients through the quantizer as if it were an identity function during backward, allowing the encoder to receive gradient signal despite the discrete bottleneck.

**Codebook collapse:** The rich-get-richer dynamic in VQ training causes a few codebook entries to dominate the argmin competition, leaving most entries with no gradient signal. Observed empirically: 13/512 active entries after 10 epochs despite reasonable reconstruction quality, because the decoder compensates for the limited codebook.

**Residual quantization:** Each RVQ stage operates on the residual of the previous stage. The residual distribution has lower variance and is more uniformly spread across the codebook space, which is why later stages achieve high utilization (69-84%) while stage 1 collapses similarly to single VQ.

## Relation to neural audio codecs

This is the same VQ/RVQ machinery used in:
- **EnCodec** (Meta) -- RVQ with K=8 codebooks at 75 Hz frame rate
- **DAC** -- improved RVQ with better codebook utilization via EMA updates
- **WavTokenizer** -- single large codebook at 40 tokens/sec for speech language modeling

The collapse mitigation used in production codecs (EMA-based codebook updates, periodic dead entry reinitialization, entropy regularization) is not implemented here -- this project intentionally leaves collapse visible to demonstrate the problem.
