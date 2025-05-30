import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

#setup 
hidden_size = 128
num_attention_heads = 16
num_key_value_heads = 4
head_dim = hidden_size // num_attention_heads
max_position_embeddings = 256
rope_theta = 10000
rms_norm_eps = 1e-5
attention_bias = False
use_qk_norm = True

# Let's give it a sample input 
batch_size = 2
sequence_length = 10
hidden_states = torch.randn(batch_size, sequence_length, hidden_size)
position_ids = torch.arange(0, sequence_length).unsqueeze(0).repeat(batch_size, 1)

attention_mask = torch.triu(torch.ones(sequence_length, sequence_length) * -torch.inf, diagonal = 1)
attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
attention_mask = attention_mask.expand(batch_size, 1, -1, -1)

print("Configuration : \n")
print(f"Hidden size: {hidden_size}")
print(f"Number of attention heads: {num_attention_heads}")
print(f" Number of Key-Value Heads: {num_key_value_heads}")
print(f"Head dims: {head_dim}")

# Setting up Q, K, V Projections 

q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias = attention_bias)
k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias = attention_bias)
v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias = attention_bias)
o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias = attention_bias)

query_states = q_proj(hidden_states)
key_states = k_proj(hidden_states)
value_states = v_proj(hidden_states)

# Need to reshare q, k, v for multi-head attention 
#target shape: (batch_size, num_heads, sequence_length, head_dim)

query_states = query_states.view(batch_size, sequence_length, num_attention_heads, head_dim).transpose(1, 2)
key_states = key_states.view(batch_size, sequence_length, num_key_value_heads, head_dim).transpose(1, 2)
value_states = value_states.view(batch_size, sequence_length, num_key_value_heads, head_dim).transpose(1, 2)


print("Projected Shapes:")
print(f"  query_states: {query_states.shape}") # (batch_size, num_attention_heads, sequence_length, head_dim)
print(f"  key_states: {key_states.shape}")     # (batch_size, num_key_value_heads, sequence_length, head_dim)
print(f"  value_states: {value_states.shape}")   # (batch_size, num_key_value_heads, sequence_length, head_dim)

num_key_value_groups = num_attention_heads // num_key_value_heads
print(f"\nNum Key/Value Groups (Q heads per K/V head): {num_key_value_groups}")

# Rotary Positional Embeddings (RoPE) - Llama models use RoPE instead of positional embeddings 
def simple_rope_calculation(dim, max_seq_len, base = 10000, device=None): 
    inv_freq = 1 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).type_as(inv_freq)
    freqs = new_func(inv_freq)
    emb = torch.cat((freqs, freqs), dim = -1)
    freqs_cos = emb.cos() 
    freqs_sin = emb.sin()
    freqs_cis = torch.complex(freqs_cos, freqs_sin)
    return freqs_cis 

def new_func(inv_freq, t): 
    freqs = torch.outer(t, inv_freq)
    return freqs 

def apply_rotary_emb_torch(
    xq: torch.Tensor,      # Query tensor, shape (batch, num_heads, seq_len, head_dim)
    xk: torch.Tensor,      # Key tensor, shape (batch, num_heads, seq_len, head_dim) - Simplified assumption
    freqs_cis: torch.Tensor, # Precomputed complex rotations, shape (max_seq_len, head_dim)
) -> Tuple[torch.Tensor, torch.Tensor]:

    freqs_cis = freqs_cis.to(xq.device)

    # 2. Select the correct rotation vectors for the current sequence positions
    #    position_ids has shape (batch, seq_len)
    #    This uses advanced indexing to pick rows from freqs_cis based on position_ids
    freqs_cis = freqs_cis[position_ids] # Now shape: (batch, seq_len, head_dim), complex

    # 3. Add a dimension for broadcasting across attention heads
    #    We want the same rotation applied to all heads for a given token/position
    freqs_cis = freqs_cis[:, None, :, :] # Now shape: (batch, 1, seq_len, head_dim), complex

    # 4. Reshape Q and K to view adjacent pairs as complex numbers
    #    xq: (batch, num_heads, seq_len, head_dim)
    #        -> reshape to (batch, num_heads, seq_len, head_dim // 2, 2)
    #        -> view as complex -> (batch, num_heads, seq_len, head_dim // 2), complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 5. Select the necessary part of freqs_cis for complex math
    #    The original freqs_cis had duplicated angles. We only need the first half
    #    (corresponding to head_dim // 2 unique frequencies) for complex multiplication.
    #    Input freqs_cis shape: (batch, 1, seq_len, head_dim), complex
    #    Output shape: (batch, 1, seq_len, head_dim // 2), complex
    freqs_cis_broadcast = freqs_cis[..., :xq_.shape[-1]] # Slices the last dim

    # --- Apply the Rotation ---

    # 6. Perform the RoPE rotation using element-wise complex multiplication
    #    xq_ (batch, num_heads, seq_len, head_dim / 2) *
    #    freqs_cis_broadcast (batch, 1, seq_len, head_dim / 2)
    #    The division by 2 is because we are treating pairs of values as complex numbers.
    #    The '1' in freqs_cis_broadcast broadcasts across the 'num_heads' dimension.
    rotated_xq = xq_ * freqs_cis_broadcast
    rotated_xk = xk_ * freqs_cis_broadcast

    # --- Convert Back to Real Representation ---

    # 7. Convert the rotated complex vectors back to real vectors
    #    rotated_xq (..., head_dim // 2) complex
    #        -> view_as_real -> (..., head_dim // 2, 2) real
    #        -> flatten last two dims -> (..., head_dim) real
    xq_out = torch.view_as_real(rotated_xq).flatten(3)
    xk_out = torch.view_as_real(rotated_xk).flatten(3)

    # 8. Cast back to the original input datatype (e.g., float16)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# Calculate RoPE frequencies (precompute usually)
# Note: RoPE is applied to head_dim, not hidden_size
freqs_cis = simple_rope_calculation(head_dim, max_position_embeddings, base=rope_theta, device=hidden_states.device)
print(f"Calculated freqs_cis shape: {freqs_cis.shape}") # (max_pos_emb, head_dim)

# Apply RoPE
# Note: RoPE is applied *before* repeating K/V for GQA
query_states_rope, key_states_rope = apply_rotary_emb_torch(query_states, key_states, freqs_cis)

print("\nShapes after RoPE:")
print(f"  query_states_rope: {query_states_rope.shape}")
print(f"  key_states_rope: {key_states_rope.shape}")

#
# Llama 4 sometimes includes an optional L2 normalization applied to Q and K *after* RoPE but *before* the attention score calculation. This is controlled by `config.use_qk_norm`.

# %%
# Llama4TextL2Norm implementation (simplified from original code)
class SimpleL2Norm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps  # Epsilon value to avoid division by zero during normalization

    def forward(self, x):
        # Normalize along the last dimension (head_dim)
        # This function normalizes the input tensor 'x' along its last dimension.
        # It computes the L2 norm (Euclidean norm) of 'x' and scales 'x' by the inverse of this norm.
        # The epsilon value is added to the denominator to ensure numerical stability and avoid division by zero.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

if use_qk_norm:
    qk_norm = SimpleL2Norm()
    query_states_final = qk_norm(query_states_rope)
    key_states_final = qk_norm(key_states_rope)
    print("\nApplied QK Norm")
else:
    query_states_final = query_states_rope
    key_states_final = key_states_rope
    print("\nSkipped QK Norm")

print("\nShapes before attention score calculation:")
print(f"  query_states_final: {query_states_final.shape}")
print(f"  key_states_final: {key_states_final.shape}")


# ## Step 4: Grouped-Query Attention (GQA) - Key/Value Repeating
#
# Since we have fewer K and V heads than Q heads (GQA), we need to "repeat" the K and V heads so that each Q head has a corresponding K and V to attend to. The `repeat_kv` function handles this.

# %%
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats Key/Value heads for GQA.
    Input: (batch, num_key_value_heads, seqlen, head_dim)
    Output: (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# Repeat K and V heads
key_states_repeated = repeat_kv(key_states_final, num_key_value_groups)
value_states_repeated = repeat_kv(value_states, num_key_value_groups) # Use original value_states, RoPE/Norm not applied to V

print("\nShapes after repeating K/V for GQA:")
print(f"  key_states_repeated: {key_states_repeated.shape}")   # Should match Q heads dimension
print(f"  value_states_repeated: {value_states_repeated.shape}") # Should match Q heads dimension

# ## Step 5: Scaled Dot-Product Attention Calculation
#
# Now we perform the standard scaled dot-product attention:
#
# \[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]
#
# 1.  **Calculate Attention Scores:** Compute the dot product between Queries (Q) and Keys (K^T).
# 2.  **Scale:** Scale the scores by \( 1/\sqrt{d_k} \) (where \(d_k\) is `head_dim`). Llama 4 also has `attn_scale` and `attn_temperature_tuning` for layers without RoPE, but we'll omit those details here.
# 3.  **Apply Mask:** Add the attention mask (`causal_mask`) to the scores. This prevents positions from attending to future positions (and optionally masks padding). Masked positions typically get a large negative value (like -inf).
# 4.  **Softmax:** Apply softmax along the key dimension to get attention weights (probabilities).
# 5.  **Dropout:** Apply dropout to the attention weights (optional, during training).
# 6.  **Calculate Output:** Compute the weighted sum of Values (V) using the attention weights.

# %%
# 1. Calculate Attention Scores (Q @ K^T)
# Q: (batch, num_attn_heads, seq_len, head_dim)
# K: (batch, num_attn_heads, seq_len, head_dim) -> K^T: (batch, num_attn_heads, head_dim, seq_len)
# Result: (batch, num_attn_heads, seq_len, seq_len)
attn_weights = torch.matmul(query_states_final, key_states_repeated.transpose(2, 3))

# 2. Scale
scaling_factor = 1.0 / math.sqrt(head_dim)
attn_weights = attn_weights * scaling_factor

# 3. Apply Mask
# Ensure mask shape is broadcastable: (batch, 1, seq_len, seq_len)
if attention_mask is not None:
    print(f"\nApplying attention mask with shape: {attention_mask.shape}")
    # Make sure mask covers the correct key length dimension
    causal_mask = attention_mask[:, :, :, :key_states_repeated.shape[-2]] # slice mask's key dim
    attn_weights = attn_weights + causal_mask
else:
     print("\nNo attention mask applied.")

# 4. Softmax
attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)

# 5. Dropout (skipped for inference example)
# attn_weights = nn.functional.dropout(attn_weights, p=attention_dropout, training=self.training)

# 6. Calculate Output (Weighted Sum of Values)
# attn_weights: (batch, num_attn_heads, seq_len, seq_len)
# V: (batch, num_attn_heads, seq_len, head_dim)
# Result: (batch, num_attn_heads, seq_len, head_dim)
attn_output = torch.matmul(attn_weights, value_states_repeated)

print("\nAttention Calculation Shapes:")
print(f"  attn_weights (raw scores): {attn_weights.shape}")
print(f"  attn_weights (after softmax): {attn_weights.shape}")
print(f"  attn_output: {attn_output.shape}")


# ## Step 6: Reshape and Output Projection
#
# Finally, the attention output heads are concatenated and passed through a final linear layer (`o_proj`) to project them back to the `hidden_size`.

# %%
# Reshape attention output
# (batch, num_attn_heads, seq_len, head_dim) -> (batch, seq_len, num_attn_heads, head_dim)
attn_output = attn_output.transpose(1, 2).contiguous()
# -> (batch, seq_len, num_attn_heads * head_dim) = (batch, seq_len, hidden_size)
attn_output = attn_output.view(batch_size, sequence_length, hidden_size)

# Apply output projection
final_attn_output = o_proj(attn_output)

print("\nFinal Output Shapes:")
print(f"  attn_output (reshaped): {attn_output.shape}")
print(f"  final_attn_output: {final_attn_output.shape}") # Should be (batch, seq_len, hidden_size)


# ## Step 7: Putting it Together (Simplified Llama4TextAttention Forward Pass)
#
# Let's combine the steps into a simplified forward function.

# %%
class SimplifiedLlama4Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_attention_heads = config['num_attention_heads']
        self.num_key_value_heads = config['num_key_value_heads']
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.max_position_embeddings = config['max_position_embeddings']
        self.rope_theta = config['rope_theta']
        self.attention_bias = config['attention_bias']
        self.use_qk_norm = config['use_qk_norm']

        if (self.head_dim * self.num_attention_heads) != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=self.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=self.attention_bias)

        self.freqs_cis = simple_rope_calculation(self.head_dim, self.max_position_embeddings, base=self.rope_theta)

        if self.use_qk_norm:
             self.qk_norm = SimpleL2Norm()

    def forward(self, hidden_states, attention_mask, position_ids):
        batch_size, sequence_length, _ = hidden_states.shape

        # Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape
        query_states = query_states.view(batch_size, sequence_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        current_freqs_cis = self.freqs_cis.to(hidden_states.device) # Get precomputed freqs
        query_states_rope, key_states_rope = apply_rotary_emb_torch(query_states, key_states, current_freqs_cis)

        # Optional QK Norm
        if self.use_qk_norm:
             query_states_final = self.qk_norm(query_states_rope)
             key_states_final = self.qk_norm(key_states_rope)
        else:
            query_states_final = query_states_rope
            key_states_final = key_states_rope


        # Repeat K/V for GQA
        key_states_repeated = repeat_kv(key_states_final, self.num_key_value_groups)
        value_states_repeated = repeat_kv(value_states, self.num_key_value_groups)

        # Attention Calculation
        attn_weights = torch.matmul(query_states_final, key_states_repeated.transpose(2, 3))
        scaling_factor = 1.0 / math.sqrt(self.head_dim)
        attn_weights = attn_weights * scaling_factor

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states_repeated.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)
        # Dropout would be here in training

        attn_output = torch.matmul(attn_weights, value_states_repeated)

        # Reshape and Output Projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, sequence_length, self.hidden_size)
        final_attn_output = self.o_proj(attn_output)

        return final_attn_output, attn_weights # Return weights for inspection


# Instantiate and run the simplified module
config_dict = {
    'hidden_size': hidden_size,
    'num_attention_heads': num_attention_heads,
    'num_key_value_heads': num_key_value_heads,
    'max_position_embeddings': max_position_embeddings,
    'rope_theta': rope_theta,
    'attention_bias': attention_bias,
    'use_qk_norm': use_qk_norm,
}

simplified_attn_module = SimplifiedLlama4Attention(config_dict)

# Run forward pass
final_output_simplified, final_weights_simplified = simplified_attn_module(hidden_states, attention_mask, position_ids)

print("\nOutput shape from simplified module:", final_output_simplified.shape)
print("Attention weights shape from simplified module:", final_weights_simplified.shape)


#
# The Llama 4 attention mechanism combines several efficient techniques:
# - **Multi-Head Attention (MHA):** Allows the model to focus on different representation subspaces.
# - **Rotary Positional Embeddings (RoPE):** Injects relative positional information effectively.
# - **Grouped-Query Attention (GQA):** Reduces the computational cost of attention by sharing Key and Value heads among Query heads.
# - **Optional QK Normalization:** Can stabilize training or improve performance.
# - **Advanced Masking:** (Not fully shown here) Handles causality, padding, and potentially chunked attention for very long sequences.
#
# These components work together to create a powerful and relatively efficient attention layer suitable for large language models.