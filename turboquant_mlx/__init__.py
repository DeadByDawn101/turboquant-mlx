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
]
