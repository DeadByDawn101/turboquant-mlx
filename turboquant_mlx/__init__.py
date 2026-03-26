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
    # Persistence (lazy-loaded)
    "get_persistent_cache",
    "get_paged_cache",
    "get_tiered_cache",
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


# ============================================================================
# Persistence & Tiered Cache (lazy-loaded)
# ============================================================================

def get_persistent_cache(**kwargs):
    """
    Get a TurboQuantCache instance for persistent KV cache save/load.
    
    Lazily imports the persistence module.
    
    Args:
        **kwargs: Arguments passed to TurboQuantCache
            - cache_dir: Directory to store cached contexts
            - bits: Quantization bits (2, 3, or 4)
            - group_size: Group size for quantization
            - compress: Whether to apply compression
            - max_cache_mb: Maximum cache size in MB
        
    Returns:
        TurboQuantCache instance
        
    Example:
        >>> cache = get_persistent_cache(bits=4)
        >>> cache.save(kv_states, "my-project", metadata={"tokens": 4096})
        >>> # Next session:
        >>> kv_states, meta = cache.load("my-project")  # 0.0003s load
    """
    from .persistence import TurboQuantCache
    return TurboQuantCache(**kwargs)


def get_paged_cache(**kwargs):
    """
    Get a PagedKVCache instance for SSD-paged context beyond GPU memory.
    
    Implements "LLM in a Flash" style paging: GPU holds recent chunks,
    older chunks are evicted to SSD and swapped back on access.
    
    Args:
        **kwargs: Arguments passed to PagedKVCache
            - max_gpu_chunks: Max chunks to keep in GPU (default: 4)
            - chunk_size: Tokens per chunk (default: 512)
            - cache_dir: Directory for SSD chunks
            - bits: Quantization bits for SSD compression
        
    Returns:
        PagedKVCache instance
        
    Example:
        >>> paged = get_paged_cache(max_gpu_chunks=4, chunk_size=512)
        >>> for chunk_id, kv_chunk in enumerate(kv_chunks):
        ...     paged.add_chunk(kv_chunk, chunk_id)
        >>> print(paged.stats)
        # {"gpu_chunks": 4, "ssd_chunks": 12, "gpu_hits": 89, "ssd_reads": 11}
    """
    from .persistence import PagedKVCache
    return PagedKVCache(**kwargs)


def get_tiered_cache(**kwargs):
    """
    Get a TieredKVCacheManager instance for GPU → SSD → R2 three-tier caching.
    
    Automatically promotes/demotes entries based on access patterns:
    - GPU: instant access, recent entries
    - SSD: 0.0003s access, compressed
    - R2: 1.5s access, cross-device sharing
    
    Args:
        **kwargs: Arguments passed to TieredKVCacheManager
            - max_gpu_mb: Max GPU tier size (default: 2000)
            - max_ssd_mb: Max SSD tier size (default: 50000)
            - r2_config: R2 config dict (endpoint, access_key, secret_key, bucket)
            - auto_promote: Auto-promote on access (default: True)
            - auto_demote: Auto-demote on memory pressure (default: True)
        
    Returns:
        TieredKVCacheManager instance
        
    Example:
        >>> manager = get_tiered_cache(max_gpu_mb=2000, max_ssd_mb=50000)
        >>> manager.put("my-context", kv_states, metadata={"tokens": 4096})
        >>> states, tier = manager.get("my-context")
        >>> print(f"Retrieved from {tier}")  # "gpu", "ssd", or "r2"
    """
    from .tiered_cache import TieredKVCacheManager
    return TieredKVCacheManager(**kwargs)
