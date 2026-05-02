# Qwen 3 From Scratch

This repository contains a lightweight PyTorch implementation of Qwen 3-style transformer components in `qwen3_from_scratch.py`.

## File: `qwen3_from_scratch.py`

This file implements the following major components:

- RoPE position encoding: `compute_rope_angles`, `apply_rope`
- RMS normalization: `RMSNorm`
- Group-query attention: `GroupQueryAttention`
- Feed-forward network: `FeedForward`
- Transformer block: `TransformerBlock`
- Full model: `Qwen3Model`
- KV cache helper: `KVCache`
- Tokenizer wrapper: `Qwen3Tokenizer`
- HF weight loader: `load_hf_weights_into_qwen`

## RoPE (Rotary Positional Embedding)

### `compute_rope_angles(head_dim, theta_base, context_length, dtype)`
- `head_dim`: even integer
- returns `cos` and `sin` tensors of shape `(context_length, head_dim // 2)`
- computes `angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)` where:
  - `positions` shape: `(context_length,)`
  - `inv_freq` shape: `(head_dim // 2,)`

### `apply_rope(x, cos, sin)`
- input `x` shape: `(B, T, num_heads, head_dim)`
- splits `x` into even and odd channels:
  - `x1 = x[..., ::2]` shape: `(B, T, num_heads, head_dim/2)`
  - `x2 = x[..., 1::2]` shape: `(B, T, num_heads, head_dim/2)`
- applies RoPE using either:
  - training mode: `cos`, `sin` shape `(T, D/2)`
  - inference mode: `cos`, `sin` shape `(D/2,)`
- recombines outputs to final shape `(B, T, num_heads, head_dim)`

## RMS Normalization

### `RMSNorm`
- normalizes input `x` across the embedding dimension
- expected input shape: `(B, T, emb_dim)`
- computes:
  - squared values: shape `(B, T, emb_dim)`
  - mean over last dim: `(B, T, 1)`
  - normalized output: `(B, T, emb_dim)`

## Group-Query Attention

### `GroupQueryAttention.__init__(d_in, num_heads, head_dim, kv_heads, dtype)`
- query projection: input `(B, T, d_in)` -> output `(B, T, num_heads * head_dim)`
- key/value projections: input `(B, T, d_in)` -> output `(B, T, kv_heads * head_dim)`
- output projection maps `(B, T, num_heads * head_dim)` back to `(B, T, d_in)`

### `GroupQueryAttention.forward(x, cos, sin, mask, cache=None)`
- input `x` shape: `(B, T, d_in)`
- after projection:
  - `query` shape: `(B, T, num_heads, head_dim)`
  - `keys_new` shape: `(B, T, kv_heads, head_dim)`
  - `values_new` shape: `(B, T, kv_heads, head_dim)`
- applies RoPE to query and keys
- transposes to:
  - `query`: `(B, num_heads, T, head_dim)`
  - `keys_new`: `(B, kv_heads, T, head_dim)`
  - `values_new`: `(B, kv_heads, T, head_dim)`
- cache handling:
  - if cached, concatenates previous and current key/value tensors along sequence dimension
  - cached shapes: `(B, kv_heads, S, head_dim)` where `S` is total stored sequence length
- repeats key/value groups to recover `num_heads`:
  - `keys`: `(B, num_heads, S, head_dim)`
  - `values`: `(B, num_heads, S, head_dim)`
- attention logits:
  - `attn_scores`: `(B, num_heads, T, S)`
  - masked and softmaxed to same shape
- context output:
  - `context`: `(B, num_heads, T, head_dim)` -> reshaped to `(B, T, num_heads * head_dim)`
  - final output after projection: `(B, T, d_in)`

## Feed-Forward Network

### `FeedForward`
- accepts `x` shape `(B, T, emb_dim)`
- projects to hidden dimension:
  - `x_fc1`: `(B, T, hidden_dim)`
  - `x_fc2`: `(B, T, hidden_dim)`
- uses SwiGLU activation:
  - `F.silu(x_fc1) * x_fc2` -> `(B, T, hidden_dim)`
- projects back to embedding dimension:
  - `out`: `(B, T, emb_dim)`

## Transformer Block

### `TransformerBlock`
- input shape: `(B, T, emb_dim)`
- applies first RMSNorm, attention, and residual connection
- then applies feed-forward, second RMSNorm, and residual connection
- output shape remains `(B, T, emb_dim)`

## Full Model: `Qwen3Model`

### Initialization
- embedding layer maps token IDs `(B, T)` to embeddings `(B, T, emb_dim)`
- stack of `n_layers` transformer blocks
- final RMSNorm and output head to logits `(B, T, vocab_size)`
- precomputes RoPE tables:
  - `cos`, `sin` shape `(context_length, head_dim)`

### Forward pass
- token IDs input: `(B, T)`
- token embeddings: `(B, T, emb_dim)`
- creates attention mask:
  - training mode: square mask `(T, T)`
  - inference mode: masked prefix with shape `(T, S)` where `S` is total past length
- broadcasted mask shape: `(1, 1, T, S)`
- each block receives:
  - `x`: `(B, T, emb_dim)`
  - `mask`: `(1, 1, T, S)`
  - `cos`, `sin` tables for RoPE
- final logits output shape: `(B, T, vocab_size)`

## KV Cache

### `KVCache`
- stores one cache entry per transformer layer
- provides:
  - `get(layer_idx)`
  - `update(layer_idx, value)`
  - `reset()`

## Tokenizer Wrapper: `Qwen3Tokenizer`

- loads tokenizer from `tokenizer.json`
- defines special tokens and maps them to IDs
- `encode(prompt)` returns a list of token IDs
- `decode(token_ids)` converts IDs back to text
- optionally wraps prompts in chat format

## HF Weight Loader

### `load_hf_weights_into_qwen(model, param_config, params)`
- assigns weights from Hugging Face parameter dicts into PyTorch model tensors
- validates shape compatibility before copying
- loads:
  - embedding weights
  - query/key/value projection weights per layer
  - output projection weights
  - normalization weights
  - feed-forward weights
- optionally ties output head to embedding weights when `lm_head.weight` is absent

## How to Use

1. Install dependencies from `requirements.txt`
2. Use `Qwen3Model(cfg)` to create the model
3. Run `forward(in_ids, cache=None)` for training or inference
4. Use `KVCache` to maintain cached key/value tensors during autoregressive decoding
5. Use `Qwen3Tokenizer` to encode/decode tokens

## Notes

- Tensor shape comments in `qwen3_from_scratch.py` are intentionally explicit for learning and debugging.
- The model is designed to match the flow of Qwen-style attention with RoPE and grouped key/value heads.
- The README describes the per-operation shapes and expected data flow through each component.

