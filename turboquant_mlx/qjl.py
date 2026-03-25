"""
QJL (Quantized Johnson-Lindenstrauss) Transform for MLX

The Johnson-Lindenstrauss lemma states that points in high-dimensional space
can be projected into lower dimensions while approximately preserving distances.

QJL quantizes this projection to 1 bit (sign only), achieving:
- Zero memory overhead (no quantization constants per block)
- Unbiased inner product estimation
- Near-optimal rate-distortion tradeoff

Key insight: For random projection matrix P, the inner product <x,y> can be
approximated using sign(Px) and sign(Py) with proper scaling.

Reference: https://dl.acm.org/doi/10.1609/aaai.v39i24.34773
"""

import mlx.core as mx
import math
from typing import Optional, Tuple


class QJLSketch:
    """
    Quantized Johnson-Lindenstrauss Sketch for inner product estimation.
    
    Projects high-dimensional vectors to lower dimensions using a random
    matrix, then quantizes to 1 bit (sign only). This enables unbiased
    inner product estimation with minimal memory overhead.
    
    Args:
        input_dim: Dimension of input vectors (e.g., head_dim)
        sketch_dim: Dimension of sketch (number of random projections)
        use_rotation: Whether to apply random rotation before projection
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        input_dim: int,
        sketch_dim: int,
        use_rotation: bool = True,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.sketch_dim = sketch_dim
        self.use_rotation = use_rotation
        self.seed = seed
        
        # Generate random projection matrix (Rademacher distribution: +1/-1)
        # Using seeded random for reproducibility
        mx.random.seed(seed)
        
        # Random projection matrix with entries +1/-1 (scaled)
        # Shape: (sketch_dim, input_dim)
        self.projection = mx.where(
            mx.random.uniform(shape=(sketch_dim, input_dim)) > 0.5,
            mx.array(1.0, dtype=mx.float32),
            mx.array(-1.0, dtype=mx.float32)
        ) / math.sqrt(sketch_dim)
        
        # Optional random rotation matrix (orthogonal via QR decomposition)
        if use_rotation:
            # Generate random matrix and orthogonalize
            random_matrix = mx.random.normal(shape=(input_dim, input_dim))
            # Use Gram-Schmidt-like orthogonalization
            self.rotation = self._orthogonalize(random_matrix)
        else:
            self.rotation = None
    
    def _orthogonalize(self, matrix: mx.array) -> mx.array:
        """
        Orthogonalize a matrix using iterative Gram-Schmidt.
        MLX doesn't have QR, so we implement a simple version.
        """
        n = matrix.shape[0]
        rows = []
        
        for i in range(n):
            v = matrix[i]
            for j in range(len(rows)):
                # Subtract projection onto previous vectors
                v = v - mx.sum(v * rows[j]) * rows[j]
            # Normalize
            norm = mx.sqrt(mx.sum(v * v) + 1e-10)
            rows.append(v / norm)
        
        return mx.stack(rows)
    
    def sketch(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Create a QJL sketch of the input tensor.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            signs: Sign bits as int8 tensor (..., sketch_dim)
            scale: Scale factor for reconstruction
        """
        original_shape = x.shape
        
        # Flatten to 2D for matmul
        x_flat = x.reshape(-1, self.input_dim)
        
        # Apply rotation if enabled
        if self.rotation is not None:
            x_flat = x_flat @ self.rotation.T
        
        # Project: (..., input_dim) @ (input_dim, sketch_dim) -> (..., sketch_dim)
        projected = x_flat @ self.projection.T
        
        # Compute scale (L2 norm of original vector)
        scale = mx.sqrt(mx.sum(x_flat * x_flat, axis=-1, keepdims=True) + 1e-10)
        
        # Quantize to sign bits (-1 or +1, stored as 0 or 1)
        signs = (projected > 0).astype(mx.int8)
        
        # Reshape back
        new_shape = original_shape[:-1] + (self.sketch_dim,)
        signs = signs.reshape(new_shape)
        scale = scale.reshape(original_shape[:-1] + (1,))
        
        return signs, scale
    
    def estimate_inner_product(
        self,
        signs_x: mx.array,
        scale_x: mx.array,
        signs_y: mx.array, 
        scale_y: mx.array,
    ) -> mx.array:
        """
        Estimate inner product from QJL sketches.
        
        The estimator: <x, y> ≈ scale_x * scale_y * (2/π) * arcsin(agreement_ratio)
        
        For high sketch_dim, this simplifies to:
        <x, y> ≈ scale_x * scale_y * (agreement_ratio - 0.5) * 2
        
        Args:
            signs_x, scale_x: QJL sketch of x
            signs_y, scale_y: QJL sketch of y
            
        Returns:
            Estimated inner product
        """
        # Convert signs from {0,1} to {-1,+1}
        sx = signs_x.astype(mx.float32) * 2 - 1
        sy = signs_y.astype(mx.float32) * 2 - 1
        
        # Agreement ratio: fraction of matching signs
        agreement = mx.mean(sx * sy, axis=-1, keepdims=True)
        
        # Scale the estimate
        # The factor (π/2) comes from the probability distribution
        # For sign agreement p, the expected cosine similarity is (2p - 1)
        estimate = scale_x * scale_y * agreement
        
        return estimate.squeeze(-1)


def qjl_compress(
    x: mx.array,
    sketch_dim: int = 256,
    seed: int = 42,
) -> Tuple[mx.array, mx.array, int]:
    """
    Compress a tensor using QJL.
    
    Args:
        x: Input tensor of shape (..., dim)
        sketch_dim: Dimension of the sketch
        seed: Random seed
        
    Returns:
        signs: Compressed sign bits
        scale: Scale factors
        seed: Seed used (for reconstruction)
    """
    dim = x.shape[-1]
    sketch = QJLSketch(dim, sketch_dim, use_rotation=True, seed=seed)
    signs, scale = sketch.sketch(x)
    return signs, scale, seed


def qjl_decompress(
    signs: mx.array,
    scale: mx.array,
    original_dim: int,
    seed: int = 42,
) -> mx.array:
    """
    Decompress QJL sketch (lossy - for debugging/visualization only).
    
    Note: QJL is designed for inner product estimation, not reconstruction.
    This provides an approximate reconstruction for testing purposes.
    
    Args:
        signs: Compressed sign bits
        scale: Scale factors  
        original_dim: Original vector dimension
        seed: Same seed used for compression
        
    Returns:
        Approximate reconstruction of original tensor
    """
    sketch_dim = signs.shape[-1]
    
    # Recreate the sketch with same seed
    mx.random.seed(seed)
    
    projection = mx.where(
        mx.random.uniform(shape=(sketch_dim, original_dim)) > 0.5,
        mx.array(1.0, dtype=mx.float32),
        mx.array(-1.0, dtype=mx.float32)
    ) / math.sqrt(sketch_dim)
    
    # Convert signs to {-1, +1}
    sx = signs.astype(mx.float32) * 2 - 1
    
    # Pseudo-inverse reconstruction (least squares)
    # x_approx = P^T @ (P @ P^T)^{-1} @ signs
    # For Rademacher, P @ P^T ≈ I, so x_approx ≈ P^T @ signs
    original_shape = signs.shape[:-1] + (original_dim,)
    signs_flat = sx.reshape(-1, sketch_dim)
    
    reconstructed = signs_flat @ projection  # (batch, original_dim)
    reconstructed = reconstructed * scale.reshape(-1, 1)
    
    return reconstructed.reshape(original_shape)


class QJLKVCompressor:
    """
    QJL compressor specifically for KV cache in attention.
    
    Compresses keys to 1-bit sketches for memory-efficient attention
    score computation.
    """
    
    def __init__(
        self,
        head_dim: int,
        sketch_dim: int = 256,
        seed: int = 42,
    ):
        self.sketch = QJLSketch(head_dim, sketch_dim, use_rotation=True, seed=seed)
        self.head_dim = head_dim
        self.sketch_dim = sketch_dim
    
    def compress_keys(self, keys: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Compress key tensors.
        
        Args:
            keys: Shape (batch, num_heads, seq_len, head_dim)
            
        Returns:
            signs: Shape (batch, num_heads, seq_len, sketch_dim) - int8
            scales: Shape (batch, num_heads, seq_len, 1) - float16
        """
        signs, scales = self.sketch.sketch(keys)
        return signs, scales.astype(mx.float16)
    
    def estimate_attention_scores(
        self,
        query: mx.array,
        key_signs: mx.array,
        key_scales: mx.array,
    ) -> mx.array:
        """
        Estimate attention scores from compressed keys.
        
        Args:
            query: Shape (batch, num_heads, 1, head_dim) - current query
            key_signs: Shape (batch, num_heads, seq_len, sketch_dim)
            key_scales: Shape (batch, num_heads, seq_len, 1)
            
        Returns:
            Estimated attention scores (batch, num_heads, 1, seq_len)
        """
        # Sketch the query
        query_signs, query_scales = self.sketch.sketch(query)
        
        # Estimate inner products
        # query: (batch, heads, 1, sketch_dim)
        # keys: (batch, heads, seq_len, sketch_dim)
        
        # Convert to {-1, +1}
        q_sx = query_signs.astype(mx.float32) * 2 - 1
        k_sx = key_signs.astype(mx.float32) * 2 - 1
        
        # Compute agreement (dot product of signs) 
        # (batch, heads, 1, sketch_dim) @ (batch, heads, sketch_dim, seq_len)
        agreement = mx.matmul(q_sx, mx.swapaxes(k_sx, -2, -1)) / self.sketch_dim
        
        # Scale by norms
        scores = agreement * query_scales * mx.swapaxes(key_scales, -2, -1)
        
        return scores
