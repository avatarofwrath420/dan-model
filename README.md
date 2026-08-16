dan-model

A decoder-only transformer implemented from scratch in PyTorch and scaled from 125M to 450M parameters. The architecture is written directly rather than assembled from transformers building blocks — the point was to understand these models by building one, not to fine-tune an existing checkpoint.

Self-directed project. No coursework requirement, no deadline.

Architecture
Component	Choice	Why
Positional encoding	RoPE (rotary)	Encodes relative position by rotation; extrapolates to longer sequences better than a learned position table
Normalization	RMSNorm	Drops mean-centering — cheaper per layer, comparable training stability
Feed-forward	SwiGLU	Gated linear unit; better loss per FLOP than a standard MLP block
Attention	Causal multi-head self-attention	Decoder-only, autoregressive

Two configurations were trained: 125M and 450M parameters. Hyperparameters for the 450M configuration were chosen by reading model specifications and the scaling literature rather than by trial and error.

Training

Training ran on rented cloud GPUs (RunPod, Google Colab), and that constraint shaped most of the engineering:

Mixed precision (bf16/fp16) with gradient accumulation — to fit a larger model and a larger effective batch size inside a fixed GPU memory budget.
Checkpointing and resume across sessions. Rented sessions terminate without warning, so a run has to survive interruption: weights, optimizer state, and step count are written to remote storage, and training restarts from the last checkpoint rather than from zero.

The second point turned out to be as much of the work as the model itself. A training run that can't survive its host disappearing is a training run you can't finish.

Status

The architecture and training pipeline are complete and both configurations have been trained. Documentation of configs, dataset details, and loss curves is still being written up.

Built by Jerry Yang · https://www.linkedin.com/in/jerry-yang-378725345/
