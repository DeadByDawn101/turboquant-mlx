"""
grove_integration.py — Bridge Grove distributed training with TurboQuant KV compression.

This module brings Grove's gradient compression techniques to KV cache compression,
enabling efficient distributed inference across Apple Silicon devices (exo clusters).

Provides:
1. SparseKVDelta: Apply SparseLoCo-style top-k delta compression to KV cache updates
   across distributed inference nodes (for exo cluster). Only transmit top-k changed
   KV vectors between nodes instead of full cache.

2. DCTKVCompressor: Apply DEMO-style DCT compression as alternative to PolarQuant.
   Transforms KV vectors to frequency domain, keeps top-k DCT components.
   
   Comparison vs PolarQuant:
   - DCT: Better for smooth/structured KV patterns, frequency-domain sparsity
   - PolarQuant: Better for general KV vectors, rotation-based quantization
   - DCT is communication-efficient (indices + values), PolarQuant is compute-efficient

3. GroveAWDLDiscovery: Use Grove's AWDL peer discovery to find exo cluster nodes
   without mDNS (solves cross-subnet discovery problem on TB4 interfaces).

Usage:
    from turboquant_mlx.grove_integration import SparseKVDelta, DCTKVCompressor
    
    # For distributed inference KV sync
    delta_compressor = SparseKVDelta(topk_ratio=0.1)
    compressed, reconstructed = delta_compressor.compress_delta(kv_new, kv_prev)
    
    # For DCT-based KV compression
    dct_comp = DCTKVCompressor(topk_components=32)
    indices, values = dct_comp.compress(kv_cache)
    kv_restored = dct_comp.decompress(indices, values, original_shape)

Author: RavenX AI / Camila Prime
License: MIT
"""

import math
from typing import Optional
import numpy as np
import mlx.core as mx


# ============================================================================
# SparseKVDelta: SparseLoCo-style top-k delta compression for KV cache sync
# ============================================================================

class SparseKVDelta:
    """
    Top-k delta compression for KV cache updates in distributed inference.
    
    Inspired by SparseLoCo gradient compression, this applies error feedback
    to ensure unsent updates accumulate and eventually get transmitted.
    
    Key insight: In distributed inference (e.g., exo cluster), nodes need to
    share KV cache updates. Instead of sending full cache, we:
    1. Compute delta = new_kv - prev_kv
    2. Keep only top-k largest magnitude changes
    3. Apply error feedback (accumulate residuals for next round)
    4. Transmit sparse (indices, values) representation
    
    This achieves ~10x compression with minimal quality loss for KV sync.
    """
    
    def __init__(
        self,
        topk_ratio: float = 0.1,
        error_decay: float = 0.95,
        chunk_size: int = 64,
    ):
        """
        Initialize SparseKVDelta compressor.
        
        Args:
            topk_ratio: Fraction of elements to keep (0.1 = 10% = 10x compression)
            error_decay: Decay factor for error feedback (0.95 = slow decay)
            chunk_size: Process in chunks for memory efficiency
        """
        self.topk_ratio = topk_ratio
        self.error_decay = error_decay
        self.chunk_size = chunk_size
        self._error_buffer: Optional[mx.array] = None
    
    def compress_delta(
        self,
        kv_new: mx.array,
        kv_prev: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Compress KV cache delta using top-k selection with error feedback.
        
        Args:
            kv_new: New KV cache state [batch, seq, heads, dim] or [seq, heads, dim]
            kv_prev: Previous KV cache state (same shape)
            
        Returns:
            (compressed_delta, reconstructed_full):
            - compressed_delta: Sparse delta (top-k selected, rest zeroed)
            - reconstructed_full: kv_prev + compressed_delta (for local attention)
        """
        # Compute raw delta
        delta = kv_new - kv_prev
        original_shape = delta.shape
        flat_delta = delta.reshape(-1)
        
        # Apply error feedback from previous round
        if self._error_buffer is not None and self._error_buffer.shape == flat_delta.shape:
            flat_delta = flat_delta + self._error_buffer * self.error_decay
        
        # Compute top-k threshold
        total_elems = flat_delta.size
        k = max(1, int(total_elems * self.topk_ratio))
        
        # Get top-k by magnitude
        abs_delta = mx.abs(flat_delta)
        
        # Use argpartition for efficient top-k selection
        if k < total_elems:
            # Get indices of top-k elements
            topk_indices = mx.argpartition(-abs_delta, kth=k)[:k]
            topk_values = flat_delta[topk_indices]
            
            # Create sparse delta (zeros except top-k)
            sparse_flat = mx.zeros_like(flat_delta)
            sparse_flat = sparse_flat.at[topk_indices].add(topk_values)
        else:
            sparse_flat = flat_delta
            topk_indices = mx.arange(total_elems)
        
        # Update error buffer with unsent residuals
        self._error_buffer = flat_delta - sparse_flat
        
        # Reshape back to original
        compressed_delta = sparse_flat.reshape(original_shape)
        reconstructed_full = kv_prev + compressed_delta
        
        mx.eval(compressed_delta, reconstructed_full, self._error_buffer)
        
        return compressed_delta, reconstructed_full
    
    def decompress_delta(
        self,
        compressed_delta: mx.array,
        kv_base: mx.array,
    ) -> mx.array:
        """
        Decompress delta and apply to base KV cache.
        
        Args:
            compressed_delta: Sparse delta from compress_delta
            kv_base: Base KV cache to add delta to
            
        Returns:
            Reconstructed KV cache (kv_base + compressed_delta)
        """
        return kv_base + compressed_delta
    
    def reset_error_buffer(self):
        """Reset error feedback buffer (call between sequences)."""
        self._error_buffer = None
    
    @property
    def compression_ratio(self) -> float:
        """Theoretical compression ratio (1/topk_ratio)."""
        return 1.0 / self.topk_ratio


# ============================================================================
# DCTKVCompressor: DEMO-style DCT compression for KV cache
# ============================================================================

def _make_dct_basis(n: int) -> np.ndarray:
    """Create DCT-II basis matrix for transform."""
    k = np.arange(n, dtype=np.float32)
    # DCT-II formula: cos(pi * k * (2*j + 1) / (2*n))
    basis = np.cos(np.pi * k[:, None] * (2 * k[None, :] + 1) / (2 * n))
    # Orthonormal scaling
    basis[0] *= 1.0 / math.sqrt(n)
    basis[1:] *= math.sqrt(2.0 / n)
    return basis


def _get_divisors(n: int) -> list[int]:
    """Get all divisors of n, sorted ascending."""
    divs = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs)


def _best_chunk(dim: int, target: int) -> int:
    """Find largest divisor of dim that's <= target."""
    for d in reversed(_get_divisors(dim)):
        if d <= target:
            return d
    return dim


class DCTKVCompressor:
    """
    DCT-based KV cache compression inspired by DeMo (Peng et al., 2024).
    
    Transforms KV vectors to frequency domain using DCT, then keeps only
    the top-k most significant frequency components. This exploits the
    observation that KV cache vectors often have smooth/structured patterns
    that compress well in frequency domain.
    
    Comparison with PolarQuant:
    - DCT: O(n log n) transform, frequency-domain sparsity, indices+values format
    - PolarQuant: O(n) rotation, magnitude+angle quantization, fixed-width format
    
    DCT is better for:
    - Distributed transmission (sparse format = fewer bytes)
    - KV caches with smooth attention patterns
    - Cases where frequency sparsity > value sparsity
    
    PolarQuant is better for:
    - Local compression (no indices overhead)
    - General KV vectors without frequency structure
    - Fixed compression ratio requirements
    """
    
    def __init__(
        self,
        topk_components: int = 32,
        chunk_size: int = 64,
    ):
        """
        Initialize DCT compressor.
        
        Args:
            topk_components: Number of DCT components to keep per chunk
            chunk_size: Size of chunks for chunked DCT (must divide vector dim)
        """
        self.topk_target = topk_components
        self.chunk_target = chunk_size
        self._basis_cache: dict[int, tuple[mx.array, mx.array]] = {}
    
    def _get_basis(self, n: int) -> tuple[mx.array, mx.array]:
        """Get or create DCT basis matrices for size n."""
        if n not in self._basis_cache:
            basis = _make_dct_basis(n)
            fwd = mx.array(basis)  # Forward DCT
            inv = mx.array(basis.T)  # Inverse DCT (transpose for orthonormal)
            self._basis_cache[n] = (fwd, inv)
        return self._basis_cache[n]
    
    def compress(self, kv: mx.array) -> tuple[mx.array, mx.array]:
        """
        Compress KV cache using chunked DCT with top-k selection.
        
        Args:
            kv: KV cache tensor of any shape (will be processed along last dim)
            
        Returns:
            (indices, values): Top-k DCT component indices and values
            Shape: indices/values are [*batch_dims, n_chunks, topk]
        """
        original_shape = kv.shape
        last_dim = original_shape[-1]
        
        # Determine chunk size (must divide last_dim)
        chunk_size = _best_chunk(last_dim, self.chunk_target)
        n_chunks = last_dim // chunk_size
        topk = min(self.topk_target, chunk_size)
        
        # Reshape to [..., n_chunks, chunk_size]
        batch_shape = original_shape[:-1]
        kv_chunked = kv.reshape(*batch_shape, n_chunks, chunk_size)
        
        # Get DCT basis
        fwd_basis, _ = self._get_basis(chunk_size)
        
        # Apply forward DCT: [..., n_chunks, chunk_size] @ [chunk_size, chunk_size]
        dct_coeffs = kv_chunked @ fwd_basis.T
        
        # Top-k selection per chunk
        if topk >= chunk_size:
            indices = mx.broadcast_to(
                mx.arange(chunk_size),
                (*batch_shape, n_chunks, chunk_size)
            )
            values = dct_coeffs
        else:
            # argpartition for top-k by magnitude
            indices = mx.argpartition(-mx.abs(dct_coeffs), kth=topk, axis=-1)[..., :topk]
            values = mx.take_along_axis(dct_coeffs, indices, axis=-1)
        
        mx.eval(indices, values)
        return indices, values
    
    def decompress(
        self,
        indices: mx.array,
        values: mx.array,
        original_shape: tuple,
    ) -> mx.array:
        """
        Decompress from top-k DCT components back to KV cache.
        
        Args:
            indices: Top-k DCT component indices from compress()
            values: Top-k DCT component values from compress()
            original_shape: Original shape of the KV cache
            
        Returns:
            Reconstructed KV cache tensor
        """
        last_dim = original_shape[-1]
        batch_shape = original_shape[:-1]
        
        # Determine chunk size
        chunk_size = _best_chunk(last_dim, self.chunk_target)
        n_chunks = last_dim // chunk_size
        topk = indices.shape[-1]
        
        # Get inverse DCT basis
        _, inv_basis = self._get_basis(chunk_size)
        
        # Reconstruct sparse DCT coefficients
        sparse_dct = mx.zeros((*batch_shape, n_chunks, chunk_size))
        sparse_dct = mx.put_along_axis(sparse_dct, indices, values, axis=-1)
        
        # Apply inverse DCT
        reconstructed = sparse_dct @ inv_basis.T
        
        # Reshape back to original
        result = reconstructed.reshape(original_shape)
        mx.eval(result)
        return result
    
    @property
    def compression_ratio(self) -> float:
        """Approximate compression ratio (chunk_size / topk_components)."""
        # Actual ratio depends on chunk_size which is data-dependent
        return self.chunk_target / self.topk_target


# ============================================================================
# GroveAWDLDiscovery: AWDL peer discovery for exo clusters
# ============================================================================

class GroveAWDLDiscovery:
    """
    AWDL peer discovery wrapper for exo cluster node discovery.
    
    Uses Grove's zero-config AWDL (AirDrop protocol) discovery to find
    nearby Apple Silicon devices without requiring mDNS or manual IPs.
    
    This solves the cross-subnet discovery problem on Thunderbolt 4 interfaces
    where mDNS doesn't propagate across network segments.
    
    Falls back to basic mDNS discovery if Grove is not installed.
    """
    
    def __init__(self):
        """Initialize discovery wrapper."""
        self._grove_available: Optional[bool] = None
        self._last_peers: list[str] = []
    
    def is_available(self) -> bool:
        """Check if Grove AWDL discovery is available."""
        if self._grove_available is None:
            try:
                import grove
                # Check if grove has the peer discovery functionality
                self._grove_available = hasattr(grove, 'init') and hasattr(grove, 'World')
            except ImportError:
                self._grove_available = False
        return self._grove_available
    
    def discover_peers(self, timeout: float = 5.0) -> list[str]:
        """
        Discover nearby peers using AWDL.
        
        Args:
            timeout: Discovery timeout in seconds
            
        Returns:
            List of peer addresses/identifiers
        """
        if not self.is_available():
            # Fallback: try basic mDNS discovery via zeroconf
            return self._mdns_fallback(timeout)
        
        try:
            import grove
            from grove.transport.p2p import P2PDiscovery
            
            # Use Grove's P2P discovery (AWDL-based)
            discovery = P2PDiscovery()
            peers = discovery.scan(timeout=timeout)
            self._last_peers = [str(p) for p in peers]
            return self._last_peers
            
        except Exception as e:
            # If Grove discovery fails, try mDNS fallback
            print(f"Grove AWDL discovery failed ({e}), falling back to mDNS")
            return self._mdns_fallback(timeout)
    
    def _mdns_fallback(self, timeout: float) -> list[str]:
        """Basic mDNS discovery fallback."""
        try:
            from zeroconf import ServiceBrowser, Zeroconf, ServiceListener
            import threading
            
            class ExoListener(ServiceListener):
                def __init__(self):
                    self.peers = []
                
                def add_service(self, zc, type_, name):
                    info = zc.get_service_info(type_, name)
                    if info:
                        for addr in info.addresses:
                            self.peers.append(f"{'.'.join(str(b) for b in addr)}:{info.port}")
                
                def remove_service(self, zc, type_, name):
                    pass
                
                def update_service(self, zc, type_, name):
                    pass
            
            zc = Zeroconf()
            listener = ExoListener()
            
            # Look for exo cluster service
            browser = ServiceBrowser(zc, "_exo._tcp.local.", listener)
            
            # Wait for discovery
            threading.Event().wait(timeout)
            
            zc.close()
            self._last_peers = listener.peers
            return self._last_peers
            
        except ImportError:
            # No zeroconf available
            return []
        except Exception:
            return []
    
    @property
    def last_discovered(self) -> list[str]:
        """Return last discovered peers without re-scanning."""
        return self._last_peers


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "SparseKVDelta",
    "DCTKVCompressor",
    "GroveAWDLDiscovery",
]
