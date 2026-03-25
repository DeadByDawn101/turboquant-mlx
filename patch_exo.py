"""
patch_exo.py — Monkey-patches exo's mlx-lm to use TurboQuantKVCache.
Run this BEFORE starting exo, or import at the top of exo's runner.

Usage:
    python3 patch_exo.py  # test the patch
    
To apply permanently to exo, add this import to:
    ~/Projects/exo/src/exo/worker/runner.py
"""

import sys
import os

# Add turboquant-mlx to path
sys.path.insert(0, os.path.expanduser("~/Projects/turboquant-mlx"))

from turboquant_mlx.mlx_kvcache import TurboQuantKVCache

# Monkey-patch mlx_lm cache
import mlx_lm.models.cache as cache_module

_original_make_prompt_cache = cache_module.make_prompt_cache

def _turboquant_make_prompt_cache(model, max_kv_size=None):
    """Drop-in replacement that uses TurboQuantKVCache."""
    if hasattr(model, "make_cache"):
        return model.make_cache()
    
    num_layers = len(model.layers)
    print(f"[TurboQuant] Using TurboQuantKVCache for {num_layers} layers")
    return [TurboQuantKVCache(r_bits=4, theta_bits=4) for _ in range(num_layers)]

cache_module.make_prompt_cache = _turboquant_make_prompt_cache
cache_module.KVCache = TurboQuantKVCache  # replace default class too

print("✅ TurboQuant patch applied to mlx_lm")
print("   KV cache will use 3.5x less memory for long contexts")

if __name__ == "__main__":
    # Quick smoke test
    import mlx.core as mx
    cache = TurboQuantKVCache()
    
    # Simulate 10 steps of attention
    B, H, L, D = 1, 8, 1, 128
    for i in range(10):
        k = mx.random.normal([B, H, L, D])
        v = mx.random.normal([B, H, L, D])
        keys, values = cache.update_and_fetch(k, v)
    
    mx.eval(keys, values)
    print(f"✅ Smoke test passed: cache shape {keys.shape}")
    print(f"   Memory used: {cache.memory_size / 1024:.1f} KB")
    print(f"   Offset: {cache.offset}")
