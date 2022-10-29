# Trader
Personal forex modeling repo

Tough but interesting goals:
- vq vae the pricing data using an SRUpp (so that token produced are still causal)
  - see [vit vq gan](https://arxiv.org/pdf/2110.04627.pdf)
    - factorized codes (i.e. just reduced dim codes before codebooking)
    - l2 normalization of codes...? seems too easy
  - Still not sure if the encoder is trainer seperately or in tandem
 
- Add denoising objectives?
  - If you already have a vq vae, wouldn't that achieve the same thing??
  - Based on [UL2](https://arxiv.org/pdf/2205.05131.pdf) and [Transcending Scaling Laws with 0.1% Extra Compute](https://arxiv.org/pdf/2210.11399.pdf)
