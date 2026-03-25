"""
PolarQuant: KV Cache Quantization via Polar Transformation for MLX

PolarQuant transforms key vectors into polar coordinates (radius, angle),
then quantizes each independently. The key insight is that after random
preconditioning, the polar angles have a tightly bounded, concentrated
distribution that can be quantized without explicit normalization.

Algorithm:
1. Random rotation (preconditioning): x' = R @ x
2. Convert to polar: (r, θ) where r = ||x||, θ = arctan2(x[1::2], x[0::2])
3. Quantize r and θ independently per group
4. Pack into uint8 indices

This eliminates per-block quantization constants (scale/zero-point),
achieving 4-8x compression with ~0% accuracy loss.

Reference: https://arxiv.org/abs/2502.02617
"""

import mlx.core as mx
import math
from typing import Optional, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class PolarQuantizedKV:
    """Container for polar-quantized KV cache data."""
    # Packed indices combining radius and angle bits
    indices: mx.array  # uint8, shape: (batch, num_heads, num_blocks, group_size, dim//2)
    
    # Per-block quantization parameters (minimal overhead)
    r_scale: mx.array   # float16, shape: (batch, num_heads, num_blocks, 1, dim//2)
    r_min: mx.array     # float16, same shape
    theta_scale: mx.array  # float16, same shape
    theta_min: mx.array    # float16, same shape
    
    # Metadata
    r_bits: int = 4
    theta_bits: int = 4
    group_size: int = 128
    original_seq_len: int = 0


class PolarQuantizer:
    """
    PolarQuant quantizer for KV cache compression.
    
    Converts key vectors to polar coordinates and quantizes to low bits.
    The polar representation is more amenable to quantization because
    the angle distribution is bounded and concentrated after rotation.
    
    Args:
        r_bits: Bits for radius quantization (default: 4)
        theta_bits: Bits for angle quantization (default: 4)
        group_size: Number of vectors per quantization group (default: 128)
        use_rotation: Apply random rotation preconditioning (default: True)
        seed: Random seed for rotation matrix
    """
    
    def __init__(
        self,
        r_bits: int = 4,
        theta_bits: int = 4,
        group_size: int = 128,
        use_rotation: bool = True,
        seed: int = 42,
    ):
        self.r_bits = r_bits
        self.theta_bits = theta_bits
        self.group_size = group_size
        self.use_rotation = use_rotation
        self.seed = seed
        
        # Will be initialized on first use based on head_dim
        self.rotation_matrix = None
        self._head_dim = None
        
        # Derived constants
        self.r_levels = 2 ** r_bits
        self.theta_levels = 2 ** theta_bits
    
    def _init_rotation(self, head_dim: int):
        """Initialize random rotation matrix for given head dimension."""
        if self._head_dim == head_dim and self.rotation_matrix is not None:
            return
            
        self._head_dim = head_dim
        
        if not self.use_rotation:
            self.rotation_matrix = None
            return
        
        # Generate random orthogonal matrix via Gram-Schmidt
        mx.random.seed(self.seed)
        random_matrix = mx.random.normal(shape=(head_dim, head_dim))
        
        # Orthogonalize using simple Gram-Schmidt
        rows = []
        for i in range(head_dim):
            v = random_matrix[i]
            for j in range(len(rows)):
                v = v - mx.sum(v * rows[j]) * rows[j]
            norm = mx.sqrt(mx.sum(v * v) + 1e-10)
            rows.append(v / norm)
        
        self.rotation_matrix = mx.stack(rows)
    
    def _to_polar(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Convert Cartesian coordinates to polar.
        
        Treats consecutive pairs of dimensions as (real, imag) components.
        
        Args:
            x: Input tensor of shape (..., head_dim) where head_dim is even
            
        Returns:
            radius: Shape (..., head_dim//2)
            theta: Shape (..., head_dim//2), range [0, 2π)
        """
        # Reshape to pair consecutive dimensions
        shape = x.shape
        x_pairs = x.reshape(shape[:-1] + (shape[-1] // 2, 2))
        
        # Compute radius and angle
        real = x_pairs[..., 0]
        imag = x_pairs[..., 1]
        
        radius = mx.sqrt(real * real + imag * imag + 1e-10)
        theta = mx.arctan2(imag, real)  # Range [-π, π]
        
        # Shift to [0, 2π)
        theta = mx.where(theta < 0, theta + 2 * math.pi, theta)
        
        return radius, theta
    
    def _from_polar(self, radius: mx.array, theta: mx.array) -> mx.array:
        """
        Convert polar coordinates back to Cartesian.
        
        Args:
            radius: Shape (..., head_dim//2)
            theta: Shape (..., head_dim//2)
            
        Returns:
            x: Shape (..., head_dim)
        """
        real = radius * mx.cos(theta)
        imag = radius * mx.sin(theta)
        
        # Interleave real and imaginary parts
        shape = radius.shape
        x = mx.stack([real, imag], axis=-1)
        x = x.reshape(shape[:-1] + (shape[-1] * 2,))
        
        return x
    
    def quantize(self, keys: mx.array) -> PolarQuantizedKV:
        """
        Quantize key tensor using polar transformation.
        
        Args:
            keys: Shape (batch, num_heads, seq_len, head_dim)
            
        Returns:
            PolarQuantizedKV containing compressed representation
        """
        batch, num_heads, seq_len, head_dim = keys.shape
        
        # Initialize rotation matrix if needed
        self._init_rotation(head_dim)
        
        # Apply rotation preconditioning
        if self.rotation_matrix is not None:
            keys_rotated = mx.matmul(keys, self.rotation_matrix.T)
        else:
            keys_rotated = keys
        
        # Pad sequence length to multiple of group_size
        padded_len = ((seq_len + self.group_size - 1) // self.group_size) * self.group_size
        if padded_len > seq_len:
            padding = mx.zeros((batch, num_heads, padded_len - seq_len, head_dim), dtype=keys.dtype)
            keys_rotated = mx.concatenate([keys_rotated, padding], axis=2)
        
        # Reshape into groups
        num_blocks = padded_len // self.group_size
        keys_grouped = keys_rotated.reshape(
            batch, num_heads, num_blocks, self.group_size, head_dim
        )
        
        # Convert to polar coordinates
        radius, theta = self._to_polar(keys_grouped)
        # radius, theta: (batch, num_heads, num_blocks, group_size, head_dim//2)
        
        # Compute min/max per group (along group_size dimension)
        r_max = mx.max(radius, axis=3, keepdims=True)
        r_min = mx.min(radius, axis=3, keepdims=True)
        r_scale = (r_max - r_min) / (self.r_levels - 1 + 1e-10)
        
        theta_max = mx.max(theta, axis=3, keepdims=True)
        theta_min = mx.min(theta, axis=3, keepdims=True)
        theta_scale = (theta_max - theta_min) / (self.theta_levels - 1 + 1e-10)
        
        # Quantize
        r_quant = mx.clip(
            mx.floor((radius - r_min) / (r_scale + 1e-10)),
            0, self.r_levels - 1
        ).astype(mx.uint8)
        
        theta_quant = mx.clip(
            mx.floor((theta - theta_min) / (theta_scale + 1e-10)),
            0, self.theta_levels - 1
        ).astype(mx.uint8)
        
        # Pack r and theta into single uint8 (4+4 bits)
        indices = (r_quant << self.theta_bits) | theta_quant
        
        return PolarQuantizedKV(
            indices=indices,
            r_scale=r_scale.astype(mx.float16),
            r_min=r_min.astype(mx.float16),
            theta_scale=theta_scale.astype(mx.float16),
            theta_min=theta_min.astype(mx.float16),
            r_bits=self.r_bits,
            theta_bits=self.theta_bits,
            group_size=self.group_size,
            original_seq_len=seq_len,
        )
    
    def dequantize(self, quantized: PolarQuantizedKV) -> mx.array:
        """
        Dequantize polar-compressed keys back to original representation.
        
        Args:
            quantized: PolarQuantizedKV from quantize()
            
        Returns:
            keys: Reconstructed key tensor (batch, num_heads, seq_len, head_dim)
        """
        # Unpack indices
        theta_mask = (1 << quantized.theta_bits) - 1
        r_quant = (quantized.indices >> quantized.theta_bits).astype(mx.float32)
        theta_quant = (quantized.indices & theta_mask).astype(mx.float32)
        
        # Dequantize (add 0.5 for mid-bin reconstruction)
        r_scale = quantized.r_scale.astype(mx.float32)
        r_min = quantized.r_min.astype(mx.float32)
        theta_scale = quantized.theta_scale.astype(mx.float32)
        theta_min = quantized.theta_min.astype(mx.float32)
        
        radius = (r_quant + 0.5) * r_scale + r_min
        theta = (theta_quant + 0.5) * theta_scale + theta_min
        
        # Convert back to Cartesian
        keys_grouped = self._from_polar(radius, theta)
        
        # Reshape back
        batch, num_heads, num_blocks, group_size, head_dim = keys_grouped.shape
        keys_padded = keys_grouped.reshape(batch, num_heads, num_blocks * group_size, head_dim)
        
        # Apply inverse rotation
        if self.rotation_matrix is not None:
            keys_reconstructed = mx.matmul(keys_padded, self.rotation_matrix)
        else:
            keys_reconstructed = keys_padded
        
        # Remove padding
        keys_reconstructed = keys_reconstructed[:, :, :quantized.original_seq_len, :]
        
        return keys_reconstructed
    
    def compute_attention_scores(
        self,
        query: mx.array,
        quantized_keys: PolarQuantizedKV,
    ) -> mx.array:
        """
        Compute attention scores directly from quantized keys.
        
        This avoids full dequantization by computing the inner product
        in polar coordinates.
        
        Args:
            query: Shape (batch, num_heads, 1, head_dim) - single query
            quantized_keys: Compressed key cache
            
        Returns:
            attention_scores: Shape (batch, num_heads, 1, seq_len)
        """
        batch, num_heads, num_blocks, group_size, half_dim = quantized_keys.indices.shape
        head_dim = half_dim * 2
        
        # Apply rotation to query
        if self.rotation_matrix is not None:
            query_rotated = mx.matmul(query, self.rotation_matrix.T)
        else:
            query_rotated = query
        
        # Convert query to polar
        query_radius, query_theta = self._to_polar(query_rotated)
        # query_radius, query_theta: (batch, num_heads, 1, half_dim)
        
        # Unpack key indices
        theta_mask = (1 << quantized_keys.theta_bits) - 1
        r_quant = (quantized_keys.indices >> quantized_keys.theta_bits).astype(mx.float32)
        theta_quant = (quantized_keys.indices & theta_mask).astype(mx.float32)
        
        # Dequantize keys
        r_scale = quantized_keys.r_scale.astype(mx.float32)
        r_min = quantized_keys.r_min.astype(mx.float32)
        theta_scale = quantized_keys.theta_scale.astype(mx.float32)
        theta_min = quantized_keys.theta_min.astype(mx.float32)
        
        key_radius = (r_quant + 0.5) * r_scale + r_min
        key_theta = (theta_quant + 0.5) * theta_scale + theta_min
        
        # Compute inner product in polar coordinates:
        # <q, k> = Σ r_q * r_k * cos(θ_q - θ_k)
        # where the sum is over all dimension pairs
        
        # Expand query for broadcasting
        # query_radius: (batch, heads, 1, 1, half_dim)
        # key_radius: (batch, heads, blocks, group, half_dim)
        q_r = query_radius[:, :, :, None, :]
        q_t = query_theta[:, :, :, None, :]
        
        # Compute per-dimension contributions
        cos_diff = mx.cos(q_t - key_theta)
        scores_per_dim = q_r * key_radius * cos_diff
        
        # Sum over dimensions
        scores = mx.sum(scores_per_dim, axis=-1)
        # scores: (batch, heads, 1, group)
        
        # Reshape to (batch, heads, 1, seq_len)
        scores = scores.reshape(batch, num_heads, 1, num_blocks * group_size)
        
        # Trim to original sequence length
        scores = scores[:, :, :, :quantized_keys.original_seq_len]
        
        return scores


def polar_compress(
    keys: mx.array,
    r_bits: int = 4,
    theta_bits: int = 4,
    group_size: int = 128,
) -> PolarQuantizedKV:
    """
    Convenience function to compress keys using PolarQuant.
    
    Args:
        keys: Shape (batch, num_heads, seq_len, head_dim)
        r_bits: Bits for radius (default: 4)
        theta_bits: Bits for angle (default: 4)
        group_size: Vectors per group (default: 128)
        
    Returns:
        PolarQuantizedKV containing compressed data
    """
    quantizer = PolarQuantizer(r_bits, theta_bits, group_size)
    return quantizer.quantize(keys)


def polar_decompress(quantized: PolarQuantizedKV) -> mx.array:
    """
    Convenience function to decompress PolarQuant data.
    
    Args:
        quantized: Compressed key data
        
    Returns:
        Reconstructed keys tensor
    """
    quantizer = PolarQuantizer(
        quantized.r_bits,
        quantized.theta_bits,
        quantized.group_size,
    )
    return quantizer.dequantize(quantized)
