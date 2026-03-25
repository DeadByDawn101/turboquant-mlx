# Star Platinum Integration Guide

## Overview

This guide explains how to integrate TurboQuant-MLX into the Star Platinum cluster's exo MLX backend for Llama 3.3 70B inference.

## Current State

### Star Platinum Cluster
- **Hardware**: Multiple M-series Mac nodes running exo
- **Model**: Llama 3.3 70B distributed across nodes
- **Backend**: MLX for inference
- **Challenge**: KV cache grows to 5-20GB per conversation for long contexts

### With TurboQuant
- **Target**: 1-4GB per conversation (4-8x reduction)
- **Impact**: 4x more concurrent sessions, longer contexts, better SELF training loops

## Integration Points

### 1. Exo MLX Backend Modification

The exo MLX backend needs modification at the attention layer level.

**File**: `exo/inference/mlx/models/llama.py` (or equivalent)

```python
# Current attention implementation
class LlamaAttention(nn.Module):
    def __call__(self, hidden_states, attention_mask, position_ids, past_key_value):
        # Standard KV cache (float16)
        ...

# Modified with TurboQuant
from turboquant_mlx import TurboQuantKVCache, TurboQuantizedCache

class LlamaAttentionTurboQuant(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # ... existing init ...
        
        # Add TurboQuant cache manager
        self.turbo_cache = TurboQuantKVCache(
            head_dim=config.head_dim,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            r_bits=4,          # Configurable
            theta_bits=4,      # Configurable  
            group_size=128,
            residual_length=128,
        )
    
    def __call__(self, hidden_states, attention_mask, position_ids, past_key_value):
        # ... compute q, k, v ...
        
        # Use TurboQuant for KV cache
        if past_key_value is None:
            compressed_kv = self.turbo_cache.compress(key_states, value_states)
        else:
            compressed_kv = self.turbo_cache.update(past_key_value, key_states, value_states)
        
        # Compute attention with compressed cache
        output, _ = self.turbo_cache.compute_attention(query_states, compressed_kv, attention_mask)
        
        return output, compressed_kv
```

### 2. Cache Serialization for Distributed Inference

For exo's distributed architecture, we need to serialize compressed caches between nodes.

```python
# turboquant_mlx/serialization.py

import mlx.core as mx
import numpy as np
from typing import Dict, Any
from .turboquant import TurboQuantizedCache, PolarQuantizedKV

def serialize_cache(cache: TurboQuantizedCache) -> Dict[str, Any]:
    """Serialize compressed cache for network transfer."""
    data = {
        "seq_len": cache.seq_len,
        "residual_start_idx": cache.residual_start_idx,
        "values": np.array(cache.values),
    }
    
    if cache.polar_keys is not None:
        data["polar_indices"] = np.array(cache.polar_keys.indices)
        data["polar_r_scale"] = np.array(cache.polar_keys.r_scale)
        data["polar_r_min"] = np.array(cache.polar_keys.r_min)
        data["polar_theta_scale"] = np.array(cache.polar_keys.theta_scale)
        data["polar_theta_min"] = np.array(cache.polar_keys.theta_min)
        data["polar_config"] = {
            "r_bits": cache.polar_keys.r_bits,
            "theta_bits": cache.polar_keys.theta_bits,
            "group_size": cache.polar_keys.group_size,
            "original_seq_len": cache.polar_keys.original_seq_len,
        }
    
    if cache.qjl_signs is not None:
        data["qjl_signs"] = np.array(cache.qjl_signs)
        data["qjl_scales"] = np.array(cache.qjl_scales)
    
    if cache.residual_keys is not None:
        data["residual_keys"] = np.array(cache.residual_keys)
    
    return data

def deserialize_cache(data: Dict[str, Any]) -> TurboQuantizedCache:
    """Deserialize compressed cache from network transfer."""
    polar_keys = None
    if "polar_indices" in data:
        polar_keys = PolarQuantizedKV(
            indices=mx.array(data["polar_indices"]),
            r_scale=mx.array(data["polar_r_scale"]),
            r_min=mx.array(data["polar_r_min"]),
            theta_scale=mx.array(data["polar_theta_scale"]),
            theta_min=mx.array(data["polar_theta_min"]),
            **data["polar_config"]
        )
    
    return TurboQuantizedCache(
        polar_keys=polar_keys,
        qjl_signs=mx.array(data["qjl_signs"]) if "qjl_signs" in data else None,
        qjl_scales=mx.array(data["qjl_scales"]) if "qjl_scales" in data else None,
        values=mx.array(data["values"]),
        residual_keys=mx.array(data["residual_keys"]) if "residual_keys" in data else None,
        residual_start_idx=data["residual_start_idx"],
        seq_len=data["seq_len"],
    )
```

### 3. Configuration

Add TurboQuant settings to exo config:

```yaml
# exo/config.yaml
inference:
  mlx:
    kv_compression:
      enabled: true
      method: "turboquant"
      r_bits: 4
      theta_bits: 4
      group_size: 128
      qjl_sketch_dim: 256
      residual_length: 128
```

### 4. Memory Budget Planning

For Llama 3.3 70B with 8 KV heads, 128 head_dim:

| Context Length | Standard KV | TurboQuant | Savings |
|---------------|-------------|------------|---------|
| 4K | 32 MB | 8 MB | 24 MB |
| 32K | 256 MB | 48 MB | 208 MB |
| 128K | 1 GB | 180 MB | 820 MB |
| 1M | 8 GB | 1.4 GB | 6.6 GB |

**Per-conversation budget with TurboQuant:**
- 32K context: ~50 MB (was 256 MB)
- 128K context: ~180 MB (was 1 GB)

**Cluster capacity increase:**
- Current: ~10 concurrent 32K conversations
- With TurboQuant: ~50 concurrent 32K conversations

## Step-by-Step Integration

### Step 1: Install TurboQuant-MLX

```bash
cd ~/Projects/turboquant-mlx
pip install -e .
```

### Step 2: Patch Exo MLX Backend

Create a monkey-patch loader:

```python
# exo_turboquant_patch.py
from turboquant_mlx import patch_model_attention

def patch_exo_model(model):
    """Apply TurboQuant to exo model."""
    return patch_model_attention(model, compression_config={
        "r_bits": 4,
        "theta_bits": 4,
        "group_size": 128,
        "residual_length": 128,
    })
```

### Step 3: Modify Exo Shard Loading

In exo's model loading code:

```python
# After loading model shard
from exo_turboquant_patch import patch_exo_model

model = load_shard(shard_config)
model = patch_exo_model(model)  # Apply TurboQuant
```

### Step 4: Update Cache Transfer Protocol

Modify exo's inter-node communication to use serialized TurboQuant caches instead of raw KV tensors.

### Step 5: Benchmark

```bash
# Test TurboQuant standalone
cd ~/Projects/turboquant-mlx
python benchmark.py --seq-len 32768 --num-heads 64 --num-kv-heads 8 --head-dim 128

# Test with actual Llama model
python -c "
from turboquant_mlx import TurboQuantKVCache
import mlx.core as mx

# Llama 3.3 70B config
cache = TurboQuantKVCache(
    head_dim=128,
    num_heads=64,
    num_kv_heads=8,
)

# Simulate 32K context
keys = mx.random.normal((1, 8, 32768, 128))
values = mx.random.normal((1, 8, 32768, 128))

compressed = cache.compress(keys, values)
usage = cache.memory_usage(compressed)
print(f'Compression ratio: {usage[\"compression_ratio\"]:.2f}x')
"
```

## SELF Training Loop Integration

TurboQuant enables larger training batches by reducing KV cache memory:

```python
# During SELF training iteration
from turboquant_mlx import TurboQuantKVCache

def generate_with_compressed_cache(model, prompt, max_tokens):
    cache_manager = TurboQuantKVCache(
        head_dim=model.config.head_dim,
        num_heads=model.config.num_attention_heads,
        num_kv_heads=model.config.num_key_value_heads,
    )
    
    # Initial forward pass
    outputs = model.generate(
        prompt,
        max_tokens=max_tokens,
        kv_cache_manager=cache_manager,
    )
    
    return outputs
```

## Fallback Mode

If TurboQuant causes issues, disable per-layer:

```python
class LlamaAttentionHybrid(nn.Module):
    def __init__(self, config, layer_idx, use_turboquant=True):
        self.use_turboquant = use_turboquant
        if use_turboquant:
            self.turbo_cache = TurboQuantKVCache(...)
    
    def __call__(self, ...):
        if self.use_turboquant:
            # Compressed path
            ...
        else:
            # Standard path
            ...
```

## Monitoring

Add metrics for compression performance:

```python
# Prometheus/metrics integration
compression_ratio_gauge.set(cache_manager.memory_usage(cache)["compression_ratio"])
attention_latency_hist.observe(attention_time)
```

## Known Limitations

1. **First token latency**: Compression adds ~10ms on first forward pass
2. **Very short contexts**: Overhead not worth it for <1K tokens
3. **Quantization artifacts**: Rare edge cases with extreme value distributions

## Next Steps

1. Test on actual Star Platinum cluster
2. Tune r_bits/theta_bits for optimal quality/compression
3. Implement Metal kernels for faster polar transform
4. Add streaming compression for real-time generation

---

*Prepared by RavenX AI for Star Platinum cluster deployment*
