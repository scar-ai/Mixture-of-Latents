# The-Latentformer
Latentformer is a transformer model with latent attention designed for efficient training. It features learnable positional embeddings, rotary position encoding, and MLA to optimize speed and performance while maintaining model quality.

## Model Architecture and Training

### Latent Attention
The model employs my interpretation of the latent attention mechanism introduced by the [deepseekv2 paper](https://arxiv.org/pdf/2405.04434). It projects queries and keys through a compressed `d_compressed`-dimensional bottleneck layer.

![Deepseek's MLA architechture](https://towardsdatascience.com/wp-content/uploads/2025/01/15MvV9YDPmc37axJe60w8Ag.png)

This design reduces computational overhead while preserving model expressiveness. The attention computation is split between two parallel paths for the Q and K channels: one processes compressed representations, while the other handles rotary position-encoded features before concatenating both paths and calculating the attention scores while the V channel is directly obtained from the compressed KV representation.

### Positional Encoding
Latentformer combines learnable positional encoding with rotary position encoding (RoPE from the [Roformer paper](https://arxiv.org/pdf/2104.09864)).
The learnable component adaptively scales and shifts sinusoidal patterns during training, allowing the model to optimize position representation.
While in the attention mechanism, RoPE provides efficient relative position handling in `d_rope` dimensions by applying rotations to the Q and K attention channels.

### Training Pipeline
The model was trained distributed training using PyTorch's DDP and mixed-precision (FP16) training.
A scheduler with warmup and step decay was used where the peak learning rate was 2.4e-4 to make converging easier on later stages of the training.
The model was trained on batches of 64 sequences of 512 tokens.

## Performance
In benchmark testing, Latentformer (set up for 348M parameters with 12 layers, 16 attention heads, 1280 embedding dimension and 256 latent dimension) achieved a validation perplexity of 36 after 2 hours and 40 minutes of training on a node of 8 AMD MI300X GPUs using the OpenWebText dataset.

## Additionnal information
- I provided with a rudimentary autoregressive training loop, however KV caching is not yet implemented.
- Use torchrun to run the DDP script: ``torchrun --nproc_per_node=<amount of GPUs in the node> main_distributed.py``
