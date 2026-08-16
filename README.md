dan-model — A decoder-only transformer built from scratch in PyTorch, scaled from 125M to 450M parameters. 
Implements RoPE positional encoding, RMSNorm, and SwiGLU feed-forward layers. 
Trained on rented cloud GPUs (RunPod, Colab) with mixed-precision (bf16/fp16) training, gradient accumulation, and checkpointing across sessions.
