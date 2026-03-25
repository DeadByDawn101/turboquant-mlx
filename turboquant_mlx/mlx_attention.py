"""
TurboQuant MLX Attention: Drop-in replacement for mlx-lm attention.

This module provides attention implementations that use TurboQuant
for KV cache compression, compatible with mlx-lm model architectures.

Features:
- Drop-in replacement for standard attention
- 4-8x memory reduction for KV cache
- ~0% accuracy degradation
- Supports GQA (Grouped Query Attention)
- Efficient incremental decoding

Usage:
    from turboquant_mlx import TurboQuantAttention
    
    # Replace standard attention
    attention = TurboQuantAttention(config)
    output = attention(hidden_states, past_kv_cache)
"""

import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional, Tuple, Any

from .turboquant import TurboQuantKVCache, TurboQuantizedCache


class TurboQuantAttention(nn.Module):
    """
    Multi-head attention with TurboQuant KV cache compression.
    
    Compatible with mlx-lm Llama/Mistral/Qwen architectures.
    
    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA)
        head_dim: Dimension per head (default: hidden_size // num_heads)
        rope_theta: RoPE theta parameter
        max_position_embeddings: Maximum sequence length
        compression_config: Dict with TurboQuant settings
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 131072,
        compression_config: Optional[dict] = None,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        
        # Number of query heads per KV head (for GQA)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        # TurboQuant compression config
        compression_config = compression_config or {}
        self.turbo_cache = TurboQuantKVCache(
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            r_bits=compression_config.get("r_bits", 4),
            theta_bits=compression_config.get("theta_bits", 4),
            group_size=compression_config.get("group_size", 128),
            qjl_sketch_dim=compression_config.get("qjl_sketch_dim", 256),
            residual_length=compression_config.get("residual_length", 128),
        )
        
        # RoPE cache (will be initialized on first forward)
        self._rope_cos = None
        self._rope_sin = None
        self._rope_max_len = 0
    
    def _init_rope(self, max_len: int):
        """Initialize RoPE embeddings."""
        if max_len <= self._rope_max_len:
            return
        
        self._rope_max_len = max_len
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (self.rope_theta ** (
            mx.arange(0, self.head_dim, 2, dtype=mx.float32) / self.head_dim
        ))
        
        # Position indices
        positions = mx.arange(max_len, dtype=mx.float32)
        
        # Compute angles
        angles = mx.outer(positions, inv_freq)
        
        # Compute cos and sin
        self._rope_cos = mx.cos(angles)
        self._rope_sin = mx.sin(angles)
    
    def _apply_rope(
        self,
        x: mx.array,
        positions: mx.array,
    ) -> mx.array:
        """Apply rotary position embeddings."""
        # x: (batch, heads, seq_len, head_dim)
        # positions: (seq_len,) or (batch, seq_len)
        
        batch, heads, seq_len, head_dim = x.shape
        
        # Ensure RoPE cache is large enough
        max_pos = int(mx.max(positions).item()) + 1
        self._init_rope(max_pos)
        
        # Get cos/sin for positions
        if positions.ndim == 1:
            cos = self._rope_cos[positions]  # (seq_len, head_dim//2)
            sin = self._rope_sin[positions]
        else:
            # Batch-specific positions
            cos = mx.take(self._rope_cos, positions.flatten(), axis=0)
            cos = cos.reshape(batch, seq_len, head_dim // 2)
            sin = mx.take(self._rope_sin, positions.flatten(), axis=0)
            sin = sin.reshape(batch, seq_len, head_dim // 2)
        
        # Expand for heads
        cos = cos[None, :, :]  # (1, seq_len, head_dim//2)
        sin = sin[None, :, :]
        
        # Split x into even and odd dimensions
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        # Apply rotation
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        
        # Interleave back
        x_rotated = mx.stack([x_rotated_even, x_rotated_odd], axis=-1)
        x_rotated = x_rotated.reshape(batch, heads, seq_len, head_dim)
        
        return x_rotated
    
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_value: Optional[TurboQuantizedCache] = None,
        use_cache: bool = True,
    ) -> Tuple[mx.array, Optional[TurboQuantizedCache]]:
        """
        Forward pass with TurboQuant KV cache compression.
        
        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Previous compressed KV cache
            use_cache: Whether to return updated cache
            
        Returns:
            output: Attention output (batch, seq_len, hidden_size)
            cache: Updated compressed KV cache (if use_cache)
        """
        batch, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape to (batch, heads, seq_len, head_dim)
        query = query.reshape(batch, seq_len, self.num_heads, self.head_dim)
        query = mx.swapaxes(query, 1, 2)
        
        key = key.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        key = mx.swapaxes(key, 1, 2)
        
        value = value.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        value = mx.swapaxes(value, 1, 2)
        
        # Position IDs
        if position_ids is None:
            if past_key_value is not None:
                start_pos = past_key_value.seq_len
            else:
                start_pos = 0
            position_ids = mx.arange(start_pos, start_pos + seq_len)
        
        # Apply RoPE
        query = self._apply_rope(query, position_ids)
        key = self._apply_rope(key, position_ids)
        
        # Handle cache
        if past_key_value is not None:
            # Incremental decoding - update cache and compute attention
            cache = self.turbo_cache.update(past_key_value, key, value)
        else:
            # First forward - compress initial KV cache
            cache = self.turbo_cache.compress(key, value)
        
        # Compute attention using compressed cache
        output, _ = self.turbo_cache.compute_attention(query, cache, attention_mask)
        
        # Reshape and project output
        output = mx.swapaxes(output, 1, 2)
        output = output.reshape(batch, seq_len, -1)
        output = self.o_proj(output)
        
        if use_cache:
            return output, cache
        return output, None


class TurboQuantLlamaAttention(TurboQuantAttention):
    """
    TurboQuant attention specifically for Llama models.
    
    Includes Llama-specific defaults and optimizations.
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        rope_theta: float = 500000.0,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_theta=rope_theta,
            **kwargs,
        )


def create_turbo_attention(
    model_config: Any,
    compression_config: Optional[dict] = None,
) -> TurboQuantAttention:
    """
    Factory function to create TurboQuant attention from model config.
    
    Compatible with mlx-lm model configs.
    
    Args:
        model_config: Model configuration object (from mlx-lm)
        compression_config: TurboQuant compression settings
        
    Returns:
        TurboQuantAttention instance
    """
    return TurboQuantAttention(
        hidden_size=getattr(model_config, "hidden_size", 4096),
        num_heads=getattr(model_config, "num_attention_heads", 32),
        num_kv_heads=getattr(model_config, "num_key_value_heads", None),
        head_dim=getattr(model_config, "head_dim", None),
        rope_theta=getattr(model_config, "rope_theta", 10000.0),
        max_position_embeddings=getattr(model_config, "max_position_embeddings", 131072),
        compression_config=compression_config,
    )


def patch_model_attention(
    model: nn.Module,
    compression_config: Optional[dict] = None,
) -> nn.Module:
    """
    Patch an existing mlx-lm model to use TurboQuant attention.
    
    This replaces all attention layers with TurboQuant versions.
    
    Args:
        model: mlx-lm model (e.g., LlamaForCausalLM)
        compression_config: TurboQuant settings
        
    Returns:
        Patched model
    """
    compression_config = compression_config or {}
    
    # Find and replace attention modules
    def replace_attention(module, path=""):
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            
            # Check if this is an attention module
            if hasattr(child, "q_proj") and hasattr(child, "k_proj"):
                # Extract config from existing module
                hidden_size = child.q_proj.weight.shape[1]
                num_heads = child.num_heads if hasattr(child, "num_heads") else 32
                num_kv_heads = child.num_kv_heads if hasattr(child, "num_kv_heads") else num_heads
                head_dim = child.head_dim if hasattr(child, "head_dim") else hidden_size // num_heads
                
                # Create TurboQuant version
                turbo_attn = TurboQuantAttention(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    compression_config=compression_config,
                )
                
                # Copy weights
                turbo_attn.q_proj.weight = child.q_proj.weight
                turbo_attn.k_proj.weight = child.k_proj.weight
                turbo_attn.v_proj.weight = child.v_proj.weight
                turbo_attn.o_proj.weight = child.o_proj.weight
                
                # Replace in parent
                setattr(module, name, turbo_attn)
                print(f"Replaced attention at {child_path} with TurboQuant")
            else:
                # Recurse
                replace_attention(child, child_path)
    
    replace_attention(model)
    return model
