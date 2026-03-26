"""
TurboQuant-MLX: KV Cache Compression for Apple Silicon

First MLX implementation of TurboQuant, achieving near-optimal rate-distortion
tradeoff for KV cache quantization with zero accuracy loss.

Based on the papers:
- TurboQuant: https://arxiv.org/abs/2504.19874
- PolarQuant: https://arxiv.org/abs/2502.02617  
- QJL: Quantized Johnson-Lindenstrauss Transform

Key Features:
- PolarQuant: Rotation-based quantization using polar coordinates
- QJL: 1-bit residual correction with Johnson-Lindenstrauss projection
- TurboQuant: Combined approach for optimal compression (4-8x reduction)
- Native MLX implementation for Apple Silicon (M-series chips)

Author: RavenX AI / DeadByDawn101
License: MIT
"""

from .qjl import QJLSketch, qjl_compress, qjl_decompress
from .polarquant import PolarQuantizer, polar_compress, polar_decompress
from .turboquant import TurboQuantKVCache, turbo_compress, turbo_decompress
from .mlx_attention import TurboQuantAttention, create_turbo_attention
from .grove_integration import SparseKVDelta, DCTKVCompressor, GroveAWDLDiscovery

__version__ = "0.1.0"
__author__ = "RavenX AI"

__all__ = [
    # QJL
    "QJLSketch",
    "qjl_compress",
    "qjl_decompress",
    # PolarQuant
    "PolarQuantizer", 
    "polar_compress",
    "polar_decompress",
    # TurboQuant
    "TurboQuantKVCache",
    "turbo_compress",
    "turbo_decompress",
    # Attention
    "TurboQuantAttention",
    "create_turbo_attention",
    # Grove Integration (SparseLoCo + DCT for distributed inference)
    "SparseKVDelta",
    "DCTKVCompressor",
    "GroveAWDLDiscovery",
    # Lazy-loaded integrations
    "get_ollama_client",
    "get_hf_cache_class",
    "patch_transformers",
]


# ============================================================================
# Optional integrations (imported lazily to avoid hard dependencies)
# ============================================================================

def get_ollama_client(**kwargs):
    """
    Get a TurboQuantOllamaClient instance.
    
    Lazily imports the Ollama integration to avoid requiring the 'openai' package
    unless this function is actually called.
    
    Args:
        **kwargs: Arguments passed to TurboQuantOllamaClient
        
    Returns:
        TurboQuantOllamaClient instance
        
    Raises:
        ImportError: If the 'openai' package is not installed
        
    Example:
        >>> client = get_ollama_client()
        >>> response = client.chat("qwen2.5:7b", messages=[...])
        >>> print(client.stats())
    """
    from .ollama_patch import TurboQuantOllamaClient
    return TurboQuantOllamaClient(**kwargs)


def get_hf_cache_class():
    """
    Get the TurboQuantHFCache class for HuggingFace transformers.
    
    Lazily imports the HF integration to avoid requiring 'transformers' 
    and 'torch' packages unless this function is actually called.
    
    Returns:
        TurboQuantHFCache class
        
    Raises:
        ImportError: If 'transformers' or 'torch' is not installed
        
    Example:
        >>> CacheClass = get_hf_cache_class()
        >>> cache = CacheClass(r_bits=4, theta_bits=4)
        >>> outputs = model.generate(**inputs, past_key_values=cache)
    """
    from .hf_patch import TurboQuantHFCache
    return TurboQuantHFCache


def patch_transformers():
    """
    Monkey-patch HuggingFace transformers to use TurboQuant by default.
    
    After calling this, all AutoModelForCausalLM.generate() calls will
    use TurboQuantHFCache unless past_key_values is explicitly provided.
    
    Lazily imports the HF integration to avoid requiring 'transformers'
    and 'torch' packages unless this function is actually called.
    
    Returns:
        True if patching succeeded, False if already patched
        
    Raises:
        ImportError: If 'transformers' or 'torch' is not installed
        
    Example:
        >>> patch_transformers()
        >>> model.generate(**inputs)  # Now uses TurboQuant compression
    """
    from .hf_patch import patch_transformers as _patch
    return _patch()
