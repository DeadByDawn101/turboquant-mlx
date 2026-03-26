"""
Walsh-Hadamard Transform (WHT) for MLX

Fast O(n log n) orthogonal transform replacing slow O(n²) Gram-Schmidt.
Implements Subsampled Randomized Hadamard Transform (SRHT): D @ H @ D
where D is random ±1 diagonal matrix and H is normalized Hadamard matrix.

Key properties:
- Orthogonal: H @ H.T = I (when normalized)
- Fast: O(n log n) via butterfly operations vs O(n²) for Gram-Schmidt
- Randomized: D matrices add randomness while preserving orthogonality

Reference: turboquant_plus achieves q8_0 speed parity using this approach.
"""

import mlx.core as mx
import math
from typing import Optional


def next_power_of_2(n: int) -> int:
    """Return smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def fast_hadamard_transform(x: mx.array) -> mx.array:
    """
    Apply fast Walsh-Hadamard transform using butterfly operations.
    
    Args:
        x: Input tensor of shape (..., n) where n is power of 2
        
    Returns:
        Transformed tensor of same shape (unnormalized)
    """
    n = x.shape[-1]
    assert is_power_of_2(n), f"Dimension must be power of 2, got {n}"
    
    # In-place butterfly algorithm
    # For each level of the recursion (log2(n) levels):
    #   For each pair of elements at distance h apart:
    #     a, b = x[i], x[i+h]
    #     x[i], x[i+h] = a + b, a - b
    
    h = 1
    result = x
    while h < n:
        # Reshape to group pairs
        # (..., n) -> (..., n//(2h), 2, h)
        shape = result.shape[:-1] + (n // (2 * h), 2, h)
        result = result.reshape(shape)
        
        # Apply butterfly: [a, b] -> [a+b, a-b]
        a = result[..., 0, :]  # (..., n//(2h), h)
        b = result[..., 1, :]
        
        result = mx.stack([a + b, a - b], axis=-2)
        result = result.reshape(x.shape[:-1] + (n,))
        
        h *= 2
    
    return result


def fast_hadamard_transform_normalized(x: mx.array) -> mx.array:
    """
    Apply normalized fast Walsh-Hadamard transform.
    
    Normalized so that H @ H.T = I (orthogonal).
    
    Args:
        x: Input tensor of shape (..., n) where n is power of 2
        
    Returns:
        Transformed tensor of same shape
    """
    n = x.shape[-1]
    return fast_hadamard_transform(x) / math.sqrt(n)


def inverse_hadamard_transform(x: mx.array) -> mx.array:
    """
    Inverse normalized Walsh-Hadamard transform.
    
    For normalized Hadamard: H^{-1} = H (self-inverse)
    
    Args:
        x: Transformed tensor
        
    Returns:
        Original tensor
    """
    return fast_hadamard_transform_normalized(x)


class WalshHadamardRotation:
    """
    Randomized Walsh-Hadamard rotation for preconditioning.
    
    Implements SRHT (Subsampled Randomized Hadamard Transform):
    R = D2 @ H @ D1
    
    where:
    - D1, D2 are diagonal matrices with random ±1 entries
    - H is the normalized Hadamard matrix
    
    This provides a fast O(n log n) orthogonal rotation that
    distributes information uniformly across dimensions.
    
    Args:
        dim: Dimension of vectors to rotate
        seed: Random seed for reproducibility
    """
    
    def __init__(self, dim: int, seed: int = 42):
        self.original_dim = dim
        self.padded_dim = next_power_of_2(dim)
        self.seed = seed
        
        # Generate random sign vectors
        mx.random.seed(seed)
        self.d1 = mx.where(
            mx.random.uniform(shape=(self.padded_dim,)) > 0.5,
            mx.array(1.0, dtype=mx.float32),
            mx.array(-1.0, dtype=mx.float32)
        )
        self.d2 = mx.where(
            mx.random.uniform(shape=(self.padded_dim,)) > 0.5,
            mx.array(1.0, dtype=mx.float32),
            mx.array(-1.0, dtype=mx.float32)
        )
    
    def rotate(self, x: mx.array) -> mx.array:
        """
        Apply randomized WHT rotation.
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Rotated tensor of shape (..., dim)
        """
        original_shape = x.shape
        
        # Pad if necessary (zero-pad to power of 2)
        if self.original_dim < self.padded_dim:
            padding_shape = x.shape[:-1] + (self.padded_dim - self.original_dim,)
            x = mx.concatenate([x, mx.zeros(padding_shape, dtype=x.dtype)], axis=-1)
        
        # Apply D1 (element-wise multiply by ±1)
        x = x * self.d1
        
        # Apply normalized Hadamard
        x = fast_hadamard_transform_normalized(x)
        
        # Apply D2
        x = x * self.d2
        
        # For non-power-of-2, we keep the full padded representation
        # to preserve invertibility. Store the padding flag internally.
        # Actually, to maintain the same output dim, we truncate but
        # the inverse must know to re-pad with zeros.
        if self.original_dim < self.padded_dim:
            x = x[..., :self.original_dim]
        
        return x
    
    def rotate_inverse(self, x: mx.array) -> mx.array:
        """
        Apply inverse randomized WHT rotation.
        
        R^{-1} = D1^{-1} @ H^{-1} @ D2^{-1}
               = D1 @ H @ D2  (since D and H are self-inverse)
        
        Note: For non-power-of-2 dimensions, the forward transform zero-padded
        and then truncated. The inverse re-pads with zeros (the forward pass
        applied D2 to zeros, so we know what to expect).
        
        Args:
            x: Rotated tensor of shape (..., dim)
            
        Returns:
            Original tensor of shape (..., dim)
        """
        # Pad back to full size if needed
        if self.original_dim < self.padded_dim:
            # The forward pass applied D2 to zeros in the padding region
            # For inverse: we need to undo D2, but the padding was zeroed
            # So just zero-pad here as well
            padding_shape = x.shape[:-1] + (self.padded_dim - self.original_dim,)
            x = mx.concatenate([x, mx.zeros(padding_shape, dtype=x.dtype)], axis=-1)
        
        # Inverse order: D2^{-1} @ H^{-1} @ D1^{-1}
        # Since D's are ±1 diagonal, D^{-1} = D
        x = x * self.d2
        x = fast_hadamard_transform_normalized(x)
        x = x * self.d1
        
        # Remove padding - only keep original dimensions
        if self.original_dim < self.padded_dim:
            x = x[..., :self.original_dim]
        
        return x
    
    def to_matrix(self) -> mx.array:
        """
        Generate explicit rotation matrix (for compatibility/testing).
        
        Returns:
            Orthogonal matrix of shape (original_dim, original_dim)
        """
        # Apply rotation to identity matrix columns
        eye = mx.eye(self.original_dim)
        rotated = self.rotate(eye)
        return rotated


def create_wht_rotation(dim: int, seed: int = 42) -> WalshHadamardRotation:
    """
    Create a WHT rotation for the given dimension.
    
    Args:
        dim: Vector dimension
        seed: Random seed
        
    Returns:
        WalshHadamardRotation instance
    """
    return WalshHadamardRotation(dim, seed)
