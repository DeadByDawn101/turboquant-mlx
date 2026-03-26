"""
TurboQuant MLX Test Suite

Tests for:
- Walsh-Hadamard Transform (WHT)
- PolarQuant quantization
- QJL sketching
- KVCache integration
- Asymmetric K/V compression
"""

import pytest
import mlx.core as mx
import math

# Test dimensions (kept small for speed)
HEAD_DIM = 64
SEQ_LEN = 128
BATCH = 1
NUM_HEADS = 4


class TestWHT:
    """Test Walsh-Hadamard Transform implementation."""
    
    def test_hadamard_power_of_2(self):
        """Test fast WHT works for power of 2 dimensions."""
        from turboquant_mlx.wht import fast_hadamard_transform_normalized
        
        x = mx.random.normal(shape=(16,))
        result = fast_hadamard_transform_normalized(x)
        
        assert result.shape == x.shape
        mx.eval(result)  # Ensure computation completes
    
    def test_hadamard_self_inverse(self):
        """Test normalized Hadamard is self-inverse: H @ H = I."""
        from turboquant_mlx.wht import fast_hadamard_transform_normalized
        
        x = mx.random.normal(shape=(32,))
        transformed = fast_hadamard_transform_normalized(x)
        recovered = fast_hadamard_transform_normalized(transformed)
        
        mx.eval(x, recovered)
        error = mx.max(mx.abs(x - recovered)).item()
        assert error < 1e-5, f"Self-inverse error too high: {error}"
    
    def test_hadamard_preserves_norm(self):
        """Test normalized WHT preserves L2 norm (orthogonal property)."""
        from turboquant_mlx.wht import fast_hadamard_transform_normalized
        
        x = mx.random.normal(shape=(64,))
        transformed = fast_hadamard_transform_normalized(x)
        
        mx.eval(x, transformed)
        original_norm = mx.sqrt(mx.sum(x * x)).item()
        transformed_norm = mx.sqrt(mx.sum(transformed * transformed)).item()
        
        assert abs(original_norm - transformed_norm) < 1e-5, \
            f"Norm changed: {original_norm} -> {transformed_norm}"
    
    def test_wht_rotation_orthogonal(self):
        """Test WHT rotation preserves inner products (orthogonality)."""
        from turboquant_mlx.wht import WalshHadamardRotation
        
        rot = WalshHadamardRotation(HEAD_DIM, seed=42)
        
        x = mx.random.normal(shape=(HEAD_DIM,))
        y = mx.random.normal(shape=(HEAD_DIM,))
        
        x_rot = rot.rotate(x)
        y_rot = rot.rotate(y)
        
        mx.eval(x, y, x_rot, y_rot)
        
        original_dot = mx.sum(x * y).item()
        rotated_dot = mx.sum(x_rot * y_rot).item()
        
        assert abs(original_dot - rotated_dot) < 1e-4, \
            f"Inner product changed: {original_dot} -> {rotated_dot}"
    
    def test_wht_rotation_invertible(self):
        """Test WHT rotation can be inverted."""
        from turboquant_mlx.wht import WalshHadamardRotation
        
        rot = WalshHadamardRotation(HEAD_DIM, seed=42)
        
        x = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        
        rotated = rot.rotate(x)
        recovered = rot.rotate_inverse(rotated)
        
        mx.eval(x, recovered)
        error = mx.max(mx.abs(x - recovered)).item()
        assert error < 1e-4, f"Inverse error too high: {error}"
    
    def test_wht_handles_non_power_of_2(self):
        """Test WHT handles non-power-of-2 dimensions via padding.
        
        Note: Non-power-of-2 dims lose some info when truncating after transform,
        but the rotation still works and preserves approximate structure.
        For best accuracy, use power-of-2 dimensions (which LLM head_dims usually are).
        """
        from turboquant_mlx.wht import WalshHadamardRotation, next_power_of_2
        
        dim = 48  # Not a power of 2
        rot = WalshHadamardRotation(dim, seed=42)
        
        x = mx.random.normal(shape=(dim,))
        rotated = rot.rotate(x)
        recovered = rot.rotate_inverse(rotated)
        
        mx.eval(x, rotated, recovered)
        
        # Shape should be preserved
        assert rotated.shape == (dim,), f"Wrong output shape: {rotated.shape}"
        
        # For non-pow2, some info is lost in truncation - this is expected
        # Just verify the transform runs and produces reasonable output
        assert not mx.any(mx.isnan(rotated)).item(), "WHT produced NaN"
        assert not mx.any(mx.isinf(rotated)).item(), "WHT produced inf"
        
        # For power of 2, should be exact inverse
        dim_pow2 = 64  # Power of 2
        rot_pow2 = WalshHadamardRotation(dim_pow2, seed=42)
        y = mx.random.normal(shape=(dim_pow2,))
        y_rot = rot_pow2.rotate(y)
        y_rec = rot_pow2.rotate_inverse(y_rot)
        mx.eval(y, y_rec)
        error_pow2 = mx.max(mx.abs(y - y_rec)).item()
        assert error_pow2 < 1e-4, f"Power-of-2 inverse error: {error_pow2}"
    
    def test_wht_batched(self):
        """Test WHT works on batched inputs."""
        from turboquant_mlx.wht import WalshHadamardRotation
        
        rot = WalshHadamardRotation(HEAD_DIM, seed=42)
        
        x = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        rotated = rot.rotate(x)
        
        mx.eval(rotated)
        assert rotated.shape == x.shape


class TestPolarQuant:
    """Test PolarQuant quantization."""
    
    def test_quantize_shape(self):
        """Test quantize produces correct output shapes."""
        from turboquant_mlx.polarquant import PolarQuantizer
        
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        quantizer = PolarQuantizer(r_bits=4, theta_bits=4, group_size=32)
        
        quantized = quantizer.quantize(keys)
        
        assert quantized.indices.dtype == mx.uint8
        assert quantized.r_scale.dtype == mx.float16
        assert quantized.original_seq_len == SEQ_LEN
    
    def test_quantize_dequantize_roundtrip(self):
        """Test quantize->dequantize roundtrip preserves approximate values."""
        from turboquant_mlx.polarquant import PolarQuantizer
        
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        quantizer = PolarQuantizer(r_bits=4, theta_bits=4, group_size=32)
        
        quantized = quantizer.quantize(keys)
        recovered = quantizer.dequantize(quantized)
        
        mx.eval(keys, recovered)
        
        # Should be within reasonable quantization error
        mse = mx.mean((keys - recovered) ** 2).item()
        assert mse < 0.5, f"MSE too high: {mse}"
        
        # Shape should match
        assert recovered.shape == keys.shape
    
    def test_compression_ratio(self):
        """Test PolarQuant achieves expected compression ratio."""
        from turboquant_mlx.polarquant import PolarQuantizer, PolarQuantizedKV
        
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        quantizer = PolarQuantizer(r_bits=4, theta_bits=4, group_size=32)
        
        quantized = quantizer.quantize(keys)
        mx.eval(quantized.indices)
        
        # Original: BATCH * NUM_HEADS * SEQ_LEN * HEAD_DIM * 4 bytes (float32)
        original_bytes = keys.size * 4
        
        # Compressed: indices (uint8) + scales (float16)
        compressed_bytes = (
            quantized.indices.size * 1 +  # uint8
            quantized.r_scale.size * 2 * 4  # float16 * 4 params
        )
        
        ratio = original_bytes / compressed_bytes
        assert ratio > 2, f"Compression ratio too low: {ratio}"
    
    def test_no_rotation_mode(self):
        """Test PolarQuant works without rotation."""
        from turboquant_mlx.polarquant import PolarQuantizer
        
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        quantizer = PolarQuantizer(use_rotation=False, group_size=32)
        
        quantized = quantizer.quantize(keys)
        recovered = quantizer.dequantize(quantized)
        
        mx.eval(keys, recovered)
        assert recovered.shape == keys.shape
    
    def test_different_bit_widths(self):
        """Test various bit width configurations."""
        from turboquant_mlx.polarquant import PolarQuantizer
        
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        
        for r_bits, theta_bits in [(2, 2), (3, 3), (4, 4), (3, 5)]:
            quantizer = PolarQuantizer(r_bits=r_bits, theta_bits=theta_bits, group_size=32)
            quantized = quantizer.quantize(keys)
            recovered = quantizer.dequantize(quantized)
            mx.eval(recovered)
            assert recovered.shape == keys.shape, f"Failed for r={r_bits}, theta={theta_bits}"


class TestQJL:
    """Test QJL (Quantized Johnson-Lindenstrauss) sketching."""
    
    def test_sketch_shape(self):
        """Test QJL sketch produces correct shapes."""
        from turboquant_mlx.qjl import QJLSketch
        
        sketch = QJLSketch(input_dim=HEAD_DIM, sketch_dim=256)
        x = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        
        signs, scale = sketch.sketch(x)
        
        assert signs.shape == (BATCH, NUM_HEADS, SEQ_LEN, 256)
        assert scale.shape == (BATCH, NUM_HEADS, SEQ_LEN, 1)
        assert signs.dtype == mx.int8
    
    def test_inner_product_estimation(self):
        """Test QJL provides reasonable inner product estimates."""
        from turboquant_mlx.qjl import QJLSketch
        
        sketch = QJLSketch(input_dim=HEAD_DIM, sketch_dim=512)
        
        x = mx.random.normal(shape=(HEAD_DIM,))
        y = mx.random.normal(shape=(HEAD_DIM,))
        
        # True inner product
        true_dot = mx.sum(x * y)
        
        # Estimated via QJL
        sx, scalex = sketch.sketch(x)
        sy, scaley = sketch.sketch(y)
        estimated_dot = sketch.estimate_inner_product(sx, scalex, sy, scaley)
        
        mx.eval(true_dot, estimated_dot)
        
        # Should be within reasonable error (QJL is approximate)
        relative_error = abs(estimated_dot.item() - true_dot.item()) / (abs(true_dot.item()) + 1e-6)
        assert relative_error < 1.0, f"Relative error too high: {relative_error}"
    
    def test_kv_compressor(self):
        """Test QJLKVCompressor interface."""
        from turboquant_mlx.qjl import QJLKVCompressor
        
        compressor = QJLKVCompressor(head_dim=HEAD_DIM, sketch_dim=256)
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        
        signs, scales = compressor.compress_keys(keys)
        
        assert signs.shape == (BATCH, NUM_HEADS, SEQ_LEN, 256)
        assert scales.dtype == mx.float16
    
    def test_attention_score_estimation(self):
        """Test QJL attention score estimation."""
        from turboquant_mlx.qjl import QJLKVCompressor
        
        compressor = QJLKVCompressor(head_dim=HEAD_DIM, sketch_dim=256)
        
        query = mx.random.normal(shape=(BATCH, NUM_HEADS, 1, HEAD_DIM))
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        
        # Compress keys
        key_signs, key_scales = compressor.compress_keys(keys)
        
        # Estimate attention scores
        estimated_scores = compressor.estimate_attention_scores(query, key_signs, key_scales)
        
        mx.eval(estimated_scores)
        assert estimated_scores.shape == (BATCH, NUM_HEADS, 1, SEQ_LEN)


class TestKVCache:
    """Test TurboQuantKVCache (drop-in mlx_lm replacement)."""
    
    def test_update_and_fetch(self):
        """Test basic update_and_fetch interface."""
        from turboquant_mlx.mlx_kvcache import TurboQuantKVCache
        
        cache = TurboQuantKVCache(
            fp16_sink_size=32,
            chunk_size=32,
            compress_after=32,
        )
        
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        values = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        
        out_k, out_v = cache.update_and_fetch(keys, values)
        
        mx.eval(out_k, out_v)
        assert out_k.shape[-2] == SEQ_LEN
        assert out_v.shape[-2] == SEQ_LEN
    
    def test_attention_sinks_fp16(self):
        """Test attention sinks are kept in fp16."""
        from turboquant_mlx.mlx_kvcache import TurboQuantKVCache
        
        cache = TurboQuantKVCache(
            fp16_sink_size=64,
            chunk_size=32,
            compress_after=64,
        )
        
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, 32, HEAD_DIM))
        values = mx.random.normal(shape=(BATCH, NUM_HEADS, 32, HEAD_DIM))
        
        cache.update_and_fetch(keys, values)
        
        # Sink should be populated and uncompressed
        assert cache._sink_keys is not None
        assert cache._sink_keys.shape[-2] == 32
        # No compression should have happened yet
        assert len(cache._comp_key_chunks) == 0
    
    def test_chunk_buffering(self):
        """Test tokens are buffered before compression."""
        from turboquant_mlx.mlx_kvcache import TurboQuantKVCache
        
        cache = TurboQuantKVCache(
            fp16_sink_size=16,
            chunk_size=32,
            compress_after=48,
        )
        
        # First batch fills sink
        k1 = mx.random.normal(shape=(BATCH, NUM_HEADS, 16, HEAD_DIM))
        v1 = mx.random.normal(shape=(BATCH, NUM_HEADS, 16, HEAD_DIM))
        cache.update_and_fetch(k1, v1)
        
        # Second batch goes to buffer
        k2 = mx.random.normal(shape=(BATCH, NUM_HEADS, 16, HEAD_DIM))
        v2 = mx.random.normal(shape=(BATCH, NUM_HEADS, 16, HEAD_DIM))
        cache.update_and_fetch(k2, v2)
        
        assert cache._buf_keys is not None
        assert cache.offset == 32
    
    def test_memory_size(self):
        """Test memory_size property returns reasonable value."""
        from turboquant_mlx.mlx_kvcache import TurboQuantKVCache
        
        cache = TurboQuantKVCache(
            fp16_sink_size=32,
            chunk_size=32,
            compress_after=32,
        )
        
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        values = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        
        cache.update_and_fetch(keys, values)
        mx.eval(cache._sink_keys)  # Force evaluation
        
        mem = cache.memory_size
        assert mem > 0, "Memory size should be positive"
    
    def test_incremental_updates(self):
        """Test incremental token additions."""
        from turboquant_mlx.mlx_kvcache import TurboQuantKVCache
        
        cache = TurboQuantKVCache(
            fp16_sink_size=32,
            chunk_size=16,
            compress_after=48,
        )
        
        total_seq = 0
        for _ in range(10):
            k = mx.random.normal(shape=(BATCH, NUM_HEADS, 8, HEAD_DIM))
            v = mx.random.normal(shape=(BATCH, NUM_HEADS, 8, HEAD_DIM))
            out_k, out_v = cache.update_and_fetch(k, v)
            total_seq += 8
            mx.eval(out_k, out_v)
            assert out_k.shape[-2] == total_seq
    
    def test_reset(self):
        """Test cache reset clears all state."""
        from turboquant_mlx.mlx_kvcache import TurboQuantKVCache
        
        cache = TurboQuantKVCache()
        
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        values = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        
        cache.update_and_fetch(keys, values)
        cache.reset()
        
        assert cache.is_empty()
        assert cache.offset == 0


class TestAsymmetric:
    """Test asymmetric K/V compression (Keys=TurboQuant, Values=PolarQuant only)."""
    
    def test_asymmetric_compression_flag(self):
        """Test use_qjl_keys=True, use_qjl_values=False (asymmetric)."""
        from turboquant_mlx.mlx_kvcache import TurboQuantKVCache
        
        cache = TurboQuantKVCache(
            use_qjl_keys=True,
            use_qjl_values=False,  # Asymmetric: QJL for keys only
            fp16_sink_size=32,
            chunk_size=32,
            compress_after=32,
        )
        
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        values = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        
        out_k, out_v = cache.update_and_fetch(keys, values)
        
        mx.eval(out_k, out_v)
        assert out_k.shape == keys.shape
        assert out_v.shape == values.shape
    
    def test_full_turboquant_both(self):
        """Test full TurboQuant on both K and V."""
        from turboquant_mlx.mlx_kvcache import TurboQuantKVCache
        
        cache = TurboQuantKVCache(
            use_qjl_keys=True,
            use_qjl_values=True,  # Both use QJL
            fp16_sink_size=32,
            chunk_size=32,
            compress_after=32,
        )
        
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        values = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        
        out_k, out_v = cache.update_and_fetch(keys, values)
        
        mx.eval(out_k, out_v)
        assert out_k.shape == keys.shape
        assert out_v.shape == values.shape
    
    def test_polar_only_mode(self):
        """Test PolarQuant only mode (no QJL)."""
        from turboquant_mlx.mlx_kvcache import TurboQuantKVCache
        
        cache = TurboQuantKVCache(
            use_qjl_keys=False,
            use_qjl_values=False,  # Pure PolarQuant
            fp16_sink_size=32,
            chunk_size=32,
            compress_after=32,
        )
        
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        values = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        
        out_k, out_v = cache.update_and_fetch(keys, values)
        
        mx.eval(out_k, out_v)
        assert out_k.shape == keys.shape
        assert out_v.shape == values.shape
    
    def test_asymmetric_preserves_accuracy(self):
        """Test asymmetric compression maintains reasonable accuracy."""
        from turboquant_mlx.mlx_kvcache import TurboQuantKVCache
        
        cache = TurboQuantKVCache(
            use_qjl_keys=True,
            use_qjl_values=False,
            fp16_sink_size=64,
            chunk_size=32,
            compress_after=64,
        )
        
        # Generate deterministic test data
        mx.random.seed(123)
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        values = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)
        
        # Check values are approximately preserved
        # (Some loss expected from PolarQuant compression)
        mse_v = mx.mean((values - out_v) ** 2).item()
        assert mse_v < 1.0, f"Value MSE too high: {mse_v}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
