
import sys
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from logger import logging
import torch.nn as nn
import torch.nn.functional as F  
import re
from tokenizers import Tokenizer
from safetensors.torch import load_file




## rope

def compute_rope_angles(head_dim, theta_base=10000, context_length=2048, dtype=torch.float32):
    
    assert head_dim % 2 == 0, "head_dim must be even"

    index = torch.arange(0, head_dim, 2, dtype=dtype)
    inv_freq = 1.0 / (theta_base ** (index / head_dim))

    ## compute positions
    positions = torch.arange(context_length, dtype=dtype) ## we are calculating the positions here  

    ## computing the angle 
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0) ### positions = 2048, 1  ** 1 ---- inv_freq = head_dim // 2 -----> angles = 2048 * 64

   
    cos = torch.cos(angles) ## 2048, 64
    sin = torch.sin(angles)  ## 2048, 64

    return  cos, sin


def apply_rope(x, cos, sin):

    B, T, H, D = x.shape

    x1 = x[..., : D // 2]   # [B, T, H, D/2]
    x2 = x[..., D // 2 :]   # [B, T, H, D/2]

    # Handle both training and inference
    if cos.dim() == 2:  # training
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, T, 1, D/2]
        sin = sin.unsqueeze(0).unsqueeze(2)
    else:               # inference (single position)
        cos = cos[None, None, None, :]       # [1, 1, 1, D/2]
        sin = sin[None, None, None, :]

    x_first = x1 * cos - x2 * sin
    x_second = x2 * cos + x1 * sin

    return torch.cat([x_first, x_second], dim=-1)  # (b, t, h, d)


## RMSNorm
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, emb, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb))

    def forward(self, x):  ## x = (b,t,emb_dim)
        square = torch.square(x)
        sq_mean = square.mean(dim=-1, keepdim=True)  #b,t,1
        value = sq_mean +  self.eps
        rms_value = torch.sqrt(value)
        normalized_value = x / rms_value
        value = normalized_value * self.weight  ## b,t,d -- ## b, t, 1 
        return value   # b, t, d


    
## group-query attention

import torch
import torch.nn as nn

class GroupQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, head_dim, kv_heads, dtype=torch.float32) :
        super().__init__()

        self.d_in = d_in
        self.num_heads = num_heads
        self.h_dim = head_dim    ## dimension per head
        self.kv_heads = kv_heads
        self.group_size = num_heads // kv_heads
        self.d_out = num_heads * self.h_dim


        self.w_query = nn.Linear(self.d_in, self.d_out,  dtype=dtype, bias=False)  ## (b, t, num_heads * h_dim)
        self.w_keys = nn.Linear(self.d_in, self.kv_heads * self.h_dim, dtype=dtype, bias=False)    ## (b, t, kv_heads * h_dim) 
        self.w_values = nn.Linear(self.d_in, self.kv_heads * self.h_dim, dtype=dtype, bias=False)  ## (b, t, kv_heads * h_dim)

        self.proj_out = nn.Linear(self.d_out, self.d_in, dtype=dtype, bias=False)   ## (b, t, num_heads * h_dim)
        

        self.q_norm = RMSNorm(self.h_dim) ## Normalize per head dimension
        self.k_norm = RMSNorm(self.h_dim)  ## Normalize per head dimension
        






    def forward(self, x, cos, sin, mask, cache=None, cache_pos=None, max_seq_len=None, is_causal=False):
        b, t, _ = x.shape

        logging.debug("GQA forward b=%s t=%s cache=%s", b, t, cache is not None)
        query = self.w_query(x)   ## x = (b, t, d_in)   -->  (b, t, num_heads * h_dim)
        keys = self.w_keys(x)    ## x = (b, t, d_in)  --> (b, t, kv_heads * h_dim)
        values = self.w_values(x)   ## x = (b, t, d_in) --> (b, t, kv_heads * h_dim)

        ## reshaping
        query = query.view(b, t, self.num_heads, self.h_dim)  ## (b, t, num_heads, h_dim)
        keys_new = keys.view(b, t, self.kv_heads, self.h_dim)    ## (b, t, kv_heads, h_dim)
        values_new = values.view(b, t, self.kv_heads, self.h_dim)   ## (b, t, kv_heads, h_dim)

        query = self.q_norm(query)
        keys_new = self.k_norm(keys_new)

        ## rope
        query = apply_rope(query, cos, sin) ##  rope expects = (b, t, num_heads, d)
        keys_new = apply_rope(keys_new, cos, sin)   ## rope expects = (b, t, kv_heads, d)

        ## reshaping --for kv cache
        query = query.transpose(1, 2)  ## (b, num_heads, t, d)
        keys_new = keys_new.transpose(1,2)  ## (b, kv_heads, t, d)
        values_new = values_new.transpose(1,2)  ## (b, kv_heads, t, d)

        if cache_pos is not None:
            pos_start, pos_end = cache_pos
            if cache is None:
                alloc_len = max(max_seq_len or pos_end, pos_end)
                cache = {
                    "keys": keys_new.new_empty(b, self.kv_heads, alloc_len, self.h_dim),
                    "values": values_new.new_empty(b, self.kv_heads, alloc_len, self.h_dim),
                }
            elif pos_end > cache["keys"].shape[2]:
                alloc_len = max(cache["keys"].shape[2] * 2, pos_end)
                new_keys = keys_new.new_empty(b, self.kv_heads, alloc_len, self.h_dim)
                new_values = values_new.new_empty(b, self.kv_heads, alloc_len, self.h_dim)
                new_keys[:, :, :pos_start, :] = cache["keys"][:, :, :pos_start, :]
                new_values[:, :, :pos_start, :] = cache["values"][:, :, :pos_start, :]
                cache = {"keys": new_keys, "values": new_values}
            keys_cache = cache["keys"]
            values_cache = cache["values"]
            keys_cache[:, :, pos_start:pos_end, :] = keys_new
            values_cache[:, :, pos_start:pos_end, :] = values_new
            keys = keys_cache[:, :, :pos_end, :]
            values = values_cache[:, :, :pos_end, :]
        else: 
            keys, values = keys_new, values_new  ## (b, kv_heads, t, d)
            
        next_cache = cache if cache is not None else (keys, values)

        if hasattr(F, "scaled_dot_product_attention"):
            # PyTorch's SDPA uses True for allowed positions; our mask uses True for blocked positions.
            context = F.scaled_dot_product_attention(
                query,
                keys,
                values,
                attn_mask=None if mask is None else ~mask,
                dropout_p=0.0,
                scale=self.h_dim ** -0.5,
                is_causal=is_causal,
                enable_gqa=self.group_size != 1,
            )
        else:
            keys = torch.repeat_interleave(keys, self.group_size, dim=1)  ## (b, num_heads, t, d)
            values = torch.repeat_interleave(values, self.group_size, dim=1)  ## (b, num_heads, t, d)
            attn_scores = query @ keys.transpose(2, 3)  ## (b, num_heads, t, total_key_tokens)
            attn_scores = attn_scores / (self.h_dim ** 0.5)
            if mask is None:
                if is_causal:
                    mask = torch.triu(
                        torch.ones(t, keys.shape[2], device=x.device, dtype=torch.bool), diagonal=1
                    )[None, None, :, :]
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask, -torch.inf)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            context = attn_weights @ values

        ## merge heads
        context = context.transpose(1, 2) ## (b, t, num_heads, d)

        context = context.reshape(b, t, self.d_out)   ## (b, t, num_heads * d)
        context = self.proj_out(context)   ## (b, t, D_model)

        return context, next_cache   ## (b, t, D_model), cache tuple


    

class FeedForward(nn.Module):
    def  __init__(self, cfg):
        super().__init__()

        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)  ## (b,t, d_model) ---> (b,t, d_ff)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)  ## (b,t, d_model)  ---> (b,t, d_ff)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)  ## (b,t, d_ff) ---> (b,t, d_model)

    def forward(self, x):
        # x = (b, t, emb_dim) or (batch, seq_len, d_model)
        x_fc1 = self.fc1(x)  ## (b, t, emb_dim) --> (b, t, d_ff)
        x_fc2 = self.fc2(x)  ## (b,t, emb_dim) --> (b,t, d_ff)

        ## swiglu activation
        x = F.silu(x_fc1) * x_fc2  ## (b,t, d_ff) * (b,t, d_ff) --> (b,t, d_ff)
        
        ## projection back 
        out = self.fc3(x)  ## (b,t, d_ff) ----> (b,t, emb_dim)

        return out  ## (b,t, emb_dim)




## transformer block 

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        ## attention
        self.att = GroupQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],            
            kv_heads=cfg["n_kv_groups"],
            dtype=cfg["dtype"]
            )
        
        self.ff = FeedForward(cfg)  ## (b,t, emb_dim)
        ## norm (b,t,D) -->  (b,t,D) ... same for both 
        self.rms_norm1 = RMSNorm(cfg["emb_dim"])
        self.rms_norm2 = RMSNorm(cfg["emb_dim"])


    def forward(self, x, mask, cos, sin, cache=None, cache_pos=None, max_seq_len=None, is_causal=False):
        ## x = (b,t,d_model)
        shortcut = x
        x = self.rms_norm1(x)  ## (b,t,D)
        x, next_cache = self.att(
            x, cos, sin, mask, cache=cache, cache_pos=cache_pos,
            max_seq_len=max_seq_len, is_causal=is_causal
        )  ##   (b,t, emb_size)
        x = x + shortcut

        ## shortcut connection for feed-forward block
        shortcut = x ### (b,t, emb_size)
        x = self.rms_norm2(x)  ## (b,t, emb_size)
        x = self.ff(x)  ## (b,t, emb_size)
        ## residual
        x = x + shortcut  ## (b,t, emb_size)

        return x, next_cache
         ### (b, t, emb_size ) and Kv tensors 






class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusable utilities
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_angles(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0  # Track current position in KV cache

    def forward(self, in_idx, cache=None):
        # Forward pass
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            if num_tokens == 1:
                mask = None
                is_causal = False
            elif pos_start == 0:
                mask = None
                is_causal = True
            else:
                query_positions = torch.arange(pos_start, pos_end, device=x.device)
                key_positions = torch.arange(pos_end, device=x.device)
                mask = key_positions[None, :] > query_positions[:, None]
                is_causal = False
        else:
            pos_start = 0  # Not strictly necessary but helps torch.compile
            mask = None
            is_causal = True
        # Prefill (no cache): mask starts as (num_tokens, num_tokens)
        # Cached decoding: mask starts as (num_tokens, prev_k_number_tokens + num_tokens)
        #
        # We add two leading dimensions so the mask becomes
        # (1, 1, num_tokens, num_tokens) during prefill and
        # (1, 1, num_tokens, total_key_tokens) during cached decoding.
        # These extra dimensions let PyTorch broadcast the same mask
        # across all batches and attention heads when applying it to
        # attn_scores of shape (batch, num_heads, num_tokens, total_key_tokens).
        if mask is not None:
            mask = mask[None, None, :, :]  # broadcast mask

        # Slice cos/sin for the current token positions
        cos = self.cos[pos_start:pos_start + num_tokens].to(x.device).to(x.dtype)
        sin = self.sin[pos_start:pos_start + num_tokens].to(x.device).to(x.dtype)

        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, cos, sin,
                                     cache=blk_cache,
                                     cache_pos=(pos_start, pos_end) if cache is not None else None,
                                     max_seq_len=cache.max_seq_len if cache is not None else None,
                                     is_causal=is_causal)
            if cache is not None:
                cache.update(i, new_blk_cache)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0







## kv cache 
class KVCache:
    def __init__(self, n_layers, max_seq_len=2048, device=None, dtype=torch.float32):
        """
        Per-layer KV cache used during autoregressive decoding.
        
        Args:
            n_layers: Number of transformer layers
            max_seq_len: Maximum sequence length metadata
            device: Device metadata for future cache allocation
            dtype: Data type metadata for future cache allocation
        """
        self.n_layers = n_layers
        self.cache = [None] * n_layers
        self.cache_len = 0  # Track current position in cache
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self._alloc_size = None  # Will be set on first use

    def get(self, layer_idx):
        return self.cache[layer_idx]
    
    def update(self, layer_idx, value):
        self.cache[layer_idx] = value
        
    def update_seq_len(self, new_len):
        """Update the current sequence length tracking"""
        self.cache_len = new_len

    def get_all(self):
        return self.cache
    
    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None
        self.cache_len = 0





def load_hf_weights_into_qwen(model, param_config, params):
    """
    Load HuggingFace model weights into Qwen3Model.
    Supports both model variants (tok_emb/trf_blocks and emb/t_block).
    """
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

        with torch.no_grad():
            if isinstance(right, torch.Tensor):
                left.copy_(right)
            else:
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

        return left

    # Handle both model attribute naming conventions
    emb_layer = model.emb if hasattr(model, 'emb') else model.tok_emb
    blocks_layer = model.t_block if hasattr(model, 't_block') else model.trf_blocks
    
    emb_layer.weight = assign(emb_layer.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for l in range(param_config["n_layers"]):  # noqa: E741
        block = blocks_layer[l]
        att = block.att

        # Q, K, V projections
        att.w_query.weight = assign(
            att.w_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        att.w_keys.weight = assign(
            att.w_keys.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        att.w_values.weight = assign(
            att.w_values.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )

        # Output projection
        att.proj_out.weight = assign(
            att.proj_out.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )

        # QK norms
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.weight = assign(
                att.q_norm.weight,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight"
            )
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.weight = assign(
                att.k_norm.weight,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight"
            )

        # Attention layernorm
        block.rms_norm1.weight = assign(
            block.rms_norm1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Feedforward weights
        if "num_experts" in param_config:
            # Load router (gating) weights
            block.ff.gate.weight = assign(
                block.ff.gate.weight,
                params[f"model.layers.{l}.mlp.gate.weight"],
                f"model.layers.{l}.mlp.gate.weight"
            )
            # Load expert weights
            for e in range(param_config["num_experts"]):
                prefix = f"model.layers.{l}.mlp.experts.{e}"
                block.ff.fc1[e].weight = assign(
                    block.ff.fc1[e].weight,
                    params[f"{prefix}.gate_proj.weight"],
                    f"{prefix}.gate_proj.weight"
                )
                block.ff.fc2[e].weight = assign(
                    block.ff.fc2[e].weight,
                    params[f"{prefix}.up_proj.weight"],
                    f"{prefix}.up_proj.weight"
                )
                block.ff.fc3[e].weight = assign(
                    block.ff.fc3[e].weight,
                    params[f"{prefix}.down_proj.weight"],
                    f"{prefix}.down_proj.weight"
                )
                # After assigning weights, move the expert layers from meta to CPU
                block.ff.fc1[e] = block.ff.fc1[e].to("cpu")
                block.ff.fc2[e] = block.ff.fc2[e].to("cpu")
                block.ff.fc3[e] = block.ff.fc3[e].to("cpu")

        else:
            block.ff.fc1.weight = assign(
                block.ff.fc1.weight,
                params[f"model.layers.{l}.mlp.gate_proj.weight"],
                f"model.layers.{l}.mlp.gate_proj.weight"
            )
            block.ff.fc2.weight = assign(
                block.ff.fc2.weight,
                params[f"model.layers.{l}.mlp.up_proj.weight"],
                f"model.layers.{l}.mlp.up_proj.weight"
            )
            block.ff.fc3.weight = assign(
                block.ff.fc3.weight,
                params[f"model.layers.{l}.mlp.down_proj.weight"],
                f"model.layers.{l}.mlp.down_proj.weight"
            )

        block.rms_norm2.weight = assign(
            block.rms_norm2.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Final normalization and output head
    model.final_norm.weight = assign(model.final_norm.weight, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params:
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = emb_layer.weight
        print("Model uses weight tying.")



# 0.6 billion parameters ## copied from sebastian raschka's notebook...
QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,     # Vocabulary size
    "context_length": 40_960,  # Length originally used during training
    "emb_dim": 1024,           # Embedding dimension
    "n_heads": 16,             # Number of attention heads
    "n_layers": 28,            # Number of layers
    "hidden_dim": 3072,        # Size of intermediate dim in FeedForward
    "head_dim": 128,           # Size of the heads in GQA
    "qk_norm": True,           # Whether to normalize queries & keys in GQA
    "n_kv_groups": 8,          # Key-Value groups for GQA
    "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
    "dtype": torch.bfloat16,   # Lower-precision dtype to reduce memory
}
