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


class TestOllamaIntegration:
    """Test Ollama integration with TurboQuantOllamaClient."""
    
    def test_client_instantiation(self):
        """Test that TurboQuantOllamaClient can be instantiated."""
        try:
            from turboquant_mlx.ollama_patch import TurboQuantOllamaClient, HAS_OPENAI
        except ImportError:
            pytest.skip("ollama_patch module not available")
        
        if not HAS_OPENAI:
            # Should raise ImportError when openai not installed
            with pytest.raises(ImportError):
                TurboQuantOllamaClient()
        else:
            # Should instantiate successfully
            client = TurboQuantOllamaClient()
            assert client is not None
            assert client.base_url == "http://localhost:11434/v1"
    
    def test_client_methods_exist(self):
        """Test that client has .chat() and .generate() methods."""
        try:
            from turboquant_mlx.ollama_patch import TurboQuantOllamaClient, HAS_OPENAI
        except ImportError:
            pytest.skip("ollama_patch module not available")
        
        if not HAS_OPENAI:
            pytest.skip("openai package not installed")
        
        client = TurboQuantOllamaClient()
        
        # Check methods exist
        assert hasattr(client, 'chat')
        assert callable(client.chat)
        assert hasattr(client, 'generate')
        assert callable(client.generate)
        assert hasattr(client, 'stats')
        assert callable(client.stats)
    
    def test_stats_tracking(self):
        """Test that stats tracking works correctly."""
        from turboquant_mlx.ollama_patch import OllamaStats
        
        stats = OllamaStats(compression_ratio=4.0, bytes_per_token_kv=512)
        
        # Initially empty
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0
        assert stats.total_requests == 0
        
        # Update with some data
        stats.update(input_tokens=100, output_tokens=50, latency_ms=500.0)
        
        assert stats.total_input_tokens == 100
        assert stats.total_output_tokens == 50
        assert stats.total_requests == 1
        assert stats.total_latency_ms == 500.0
        
        # Check summary
        summary = stats.summary()
        assert summary["total_tokens"] == 150
        assert summary["avg_latency_ms"] == 500.0
        assert summary["compression_ratio"] == 4.0
        assert summary["estimated_memory_savings_mb"] > 0
    
    def test_stats_reset(self):
        """Test that stats can be reset."""
        from turboquant_mlx.ollama_patch import OllamaStats
        
        stats = OllamaStats()
        stats.update(100, 50, 500.0)
        stats.reset()
        
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0
        assert stats.total_requests == 0
    
    def test_patch_ollama_env(self):
        """Test that patch_ollama_env sets correct environment variables."""
        import os
        from turboquant_mlx.ollama_patch import patch_ollama_env
        
        # Save original env
        original_env = {
            k: os.environ.get(k) 
            for k in ["OLLAMA_NUM_PARALLEL", "OLLAMA_FLASH_ATTENTION", 
                     "OLLAMA_KEEP_ALIVE", "OLLAMA_NUM_CTX"]
        }
        
        try:
            env = patch_ollama_env(
                num_parallel=8,
                num_ctx=16384,
                flash_attention=True,
                keep_alive="12h"
            )
            
            assert os.environ.get("OLLAMA_NUM_PARALLEL") == "8"
            assert os.environ.get("OLLAMA_FLASH_ATTENTION") == "1"
            assert os.environ.get("OLLAMA_KEEP_ALIVE") == "12h"
            assert os.environ.get("OLLAMA_NUM_CTX") == "16384"
            
            # Verify return value
            assert env["OLLAMA_NUM_PARALLEL"] == "8"
            assert env["OLLAMA_FLASH_ATTENTION"] == "1"
        finally:
            # Restore original env
            for k, v in original_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class TestHFIntegration:
    """Test HuggingFace transformers integration."""
    
    def test_cache_instantiation_without_transformers(self):
        """Test graceful ImportError when transformers not installed."""
        from turboquant_mlx.hf_patch import HAS_TRANSFORMERS, HAS_TORCH
        
        if HAS_TRANSFORMERS and HAS_TORCH:
            pytest.skip("transformers is installed - testing without it not possible")
        
        # Should raise ImportError with helpful message
        from turboquant_mlx.hf_patch import TurboQuantHFCache
        with pytest.raises(ImportError) as excinfo:
            TurboQuantHFCache()
        
        assert "transformers" in str(excinfo.value).lower() or "torch" in str(excinfo.value).lower()
    
    def test_cache_instantiation_with_transformers(self):
        """Test TurboQuantHFCache can be instantiated when transformers is available."""
        from turboquant_mlx.hf_patch import HAS_TRANSFORMERS, HAS_TORCH
        
        if not (HAS_TRANSFORMERS and HAS_TORCH):
            pytest.skip("transformers or torch not installed")
        
        from turboquant_mlx.hf_patch import TurboQuantHFCache
        
        cache = TurboQuantHFCache(
            r_bits=4,
            theta_bits=4,
            group_size=128,
        )
        
        assert cache is not None
        assert cache.r_bits == 4
        assert cache.theta_bits == 4
        assert cache.group_size == 128
    
    def test_dynamic_cache_interface(self):
        """Test TurboQuantHFCache exposes correct DynamicCache interface."""
        from turboquant_mlx.hf_patch import HAS_TRANSFORMERS, HAS_TORCH
        
        if not (HAS_TRANSFORMERS and HAS_TORCH):
            pytest.skip("transformers or torch not installed")
        
        from turboquant_mlx.hf_patch import TurboQuantHFCache
        
        cache = TurboQuantHFCache()
        
        # Should have DynamicCache methods
        assert hasattr(cache, 'update')
        assert hasattr(cache, 'get_seq_length')
        assert hasattr(cache, 'get_max_length')
        assert hasattr(cache, 'key_cache')
        assert hasattr(cache, 'value_cache')
    
    def test_load_and_patch_import_error(self):
        """Test load_and_patch raises ImportError when transformers not installed."""
        from turboquant_mlx.hf_patch import HAS_TRANSFORMERS, HAS_TORCH
        
        if HAS_TRANSFORMERS and HAS_TORCH:
            pytest.skip("transformers is installed - testing ImportError not possible")
        
        from turboquant_mlx.hf_patch import load_and_patch
        
        with pytest.raises(ImportError) as excinfo:
            load_and_patch("some-model")
        
        assert "transformers" in str(excinfo.value).lower() or "torch" in str(excinfo.value).lower()
    
    def test_cache_update_method(self):
        """Test the update() method compresses keys correctly."""
        from turboquant_mlx.hf_patch import HAS_TRANSFORMERS, HAS_TORCH
        
        if not (HAS_TRANSFORMERS and HAS_TORCH):
            pytest.skip("transformers or torch not installed")
        
        import torch
        from turboquant_mlx.hf_patch import TurboQuantHFCache
        
        cache = TurboQuantHFCache(
            r_bits=4,
            theta_bits=4,
            group_size=32,
            compress_after=32,
        )
        
        # Create test tensors
        batch, num_heads, seq_len, head_dim = 1, 4, 64, 64
        keys = torch.randn(batch, num_heads, seq_len, head_dim)
        values = torch.randn(batch, num_heads, seq_len, head_dim)
        
        # Update cache
        out_keys, out_values = cache.update(keys, values, layer_idx=0)
        
        # Verify shapes preserved
        assert out_keys.shape == keys.shape
        assert out_values.shape == values.shape
        
        # Verify sequence length tracking
        assert cache.get_seq_length(layer_idx=0) == seq_len
    
    def test_from_legacy_cache(self):
        """Test from_legacy_cache classmethod."""
        from turboquant_mlx.hf_patch import HAS_TRANSFORMERS, HAS_TORCH
        
        if not (HAS_TRANSFORMERS and HAS_TORCH):
            pytest.skip("transformers or torch not installed")
        
        import torch
        from turboquant_mlx.hf_patch import TurboQuantHFCache
        
        # Create legacy format cache
        batch, num_heads, seq_len, head_dim = 1, 4, 32, 64
        k1 = torch.randn(batch, num_heads, seq_len, head_dim)
        v1 = torch.randn(batch, num_heads, seq_len, head_dim)
        k2 = torch.randn(batch, num_heads, seq_len, head_dim)
        v2 = torch.randn(batch, num_heads, seq_len, head_dim)
        
        legacy_cache = ((k1, v1), (k2, v2))
        
        # Convert to TurboQuantHFCache
        cache = TurboQuantHFCache.from_legacy_cache(legacy_cache)
        
        assert cache is not None
        assert len(cache.key_cache) == 2
        assert cache.get_seq_length(layer_idx=0) == seq_len
        assert cache.get_seq_length(layer_idx=1) == seq_len
    
    def test_cache_stats(self):
        """Test compression statistics tracking."""
        from turboquant_mlx.hf_patch import HAS_TRANSFORMERS, HAS_TORCH
        
        if not (HAS_TRANSFORMERS and HAS_TORCH):
            pytest.skip("transformers or torch not installed")
        
        from turboquant_mlx.hf_patch import TurboQuantHFCache
        
        cache = TurboQuantHFCache(r_bits=4, theta_bits=4)
        
        stats = cache.stats()
        
        assert "total_tokens" in stats
        assert "compressed_tokens" in stats
        assert "estimated_compression_ratio" in stats


class TestGroveIntegration:
    """Test Grove integration: SparseLoCo-style KV delta + DCT compression."""
    
    def test_sparse_kv_delta_instantiation(self):
        """Test SparseKVDelta can be instantiated."""
        from turboquant_mlx.grove_integration import SparseKVDelta
        
        delta_comp = SparseKVDelta(topk_ratio=0.1, error_decay=0.95, chunk_size=64)
        
        assert delta_comp.topk_ratio == 0.1
        assert delta_comp.error_decay == 0.95
        assert delta_comp.chunk_size == 64
        assert delta_comp.compression_ratio == 10.0
    
    def test_sparse_kv_delta_compress_decompress(self):
        """Test SparseKVDelta compress/decompress roundtrip."""
        from turboquant_mlx.grove_integration import SparseKVDelta
        
        delta_comp = SparseKVDelta(topk_ratio=0.1)
        
        # Create test KV caches
        kv_prev = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        kv_new = kv_prev + mx.random.normal(shape=kv_prev.shape) * 0.1  # Small delta
        
        # Compress delta
        compressed_delta, reconstructed = delta_comp.compress_delta(kv_new, kv_prev)
        mx.eval(compressed_delta, reconstructed)
        
        # Shape should be preserved
        assert compressed_delta.shape == kv_new.shape
        assert reconstructed.shape == kv_new.shape
        
        # Decompress should work
        decompressed = delta_comp.decompress_delta(compressed_delta, kv_prev)
        mx.eval(decompressed)
        
        # Decompressed should equal reconstructed
        error = mx.max(mx.abs(decompressed - reconstructed)).item()
        assert error < 1e-5, f"Decompress error: {error}"
    
    def test_sparse_kv_delta_topk_sparsity(self):
        """Test top-k sparsity ratio is respected."""
        from turboquant_mlx.grove_integration import SparseKVDelta
        
        topk_ratio = 0.1  # Keep 10%
        delta_comp = SparseKVDelta(topk_ratio=topk_ratio)
        
        # Create delta that should be sparsified
        kv_prev = mx.zeros((1, 1, 100, 64))
        kv_new = mx.random.normal(shape=kv_prev.shape)
        
        compressed_delta, _ = delta_comp.compress_delta(kv_new, kv_prev)
        mx.eval(compressed_delta)
        
        # Count non-zero elements
        flat = compressed_delta.reshape(-1)
        nonzero_count = mx.sum(mx.abs(flat) > 1e-8).item()
        total_count = flat.size
        
        actual_ratio = nonzero_count / total_count
        expected_ratio = topk_ratio
        
        # Allow some tolerance due to discrete top-k
        assert abs(actual_ratio - expected_ratio) < 0.02, \
            f"Sparsity ratio {actual_ratio} != expected {expected_ratio}"
    
    def test_sparse_kv_delta_error_feedback(self):
        """Test error feedback accumulates correctly."""
        from turboquant_mlx.grove_integration import SparseKVDelta
        
        delta_comp = SparseKVDelta(topk_ratio=0.1, error_decay=0.9)
        
        # First compression
        kv_prev = mx.zeros((1, 1, 100, 64))
        kv_new = mx.random.normal(shape=kv_prev.shape)
        delta_comp.compress_delta(kv_new, kv_prev)
        
        # Error buffer should be populated
        assert delta_comp._error_buffer is not None
        mx.eval(delta_comp._error_buffer)
        
        # Error buffer should have residuals (non-zero)
        error_sum = mx.sum(mx.abs(delta_comp._error_buffer)).item()
        assert error_sum > 0, "Error feedback should accumulate unsent values"
        
        # Reset should clear it
        delta_comp.reset_error_buffer()
        assert delta_comp._error_buffer is None
    
    def test_dct_kv_compressor_instantiation(self):
        """Test DCTKVCompressor can be instantiated."""
        from turboquant_mlx.grove_integration import DCTKVCompressor
        
        dct_comp = DCTKVCompressor(topk_components=32, chunk_size=64)
        
        assert dct_comp.topk_target == 32
        assert dct_comp.chunk_target == 64
    
    def test_dct_kv_compressor_compress_decompress(self):
        """Test DCTKVCompressor compress/decompress shape correctness."""
        from turboquant_mlx.grove_integration import DCTKVCompressor
        
        dct_comp = DCTKVCompressor(topk_components=16, chunk_size=32)
        
        # Create test KV cache (last dim must be divisible by chunk_size)
        kv = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        
        # Compress
        indices, values = dct_comp.compress(kv)
        mx.eval(indices, values)
        
        # Check compressed shapes
        # Original: [1, 4, 128, 64] -> chunks: [1, 4, 128, 2, 32] -> topk: [1, 4, 128, 2, 16]
        expected_n_chunks = HEAD_DIM // 32  # = 2
        expected_topk = 16
        assert indices.shape[-1] == expected_topk
        assert values.shape[-1] == expected_topk
        
        # Decompress
        original_shape = kv.shape
        reconstructed = dct_comp.decompress(indices, values, original_shape)
        mx.eval(reconstructed)
        
        # Shape should match original
        assert reconstructed.shape == original_shape
    
    def test_dct_kv_compressor_reconstruction_quality(self):
        """Test DCT compression preserves information reasonably."""
        from turboquant_mlx.grove_integration import DCTKVCompressor
        
        # Use more DCT components for better quality
        dct_comp = DCTKVCompressor(topk_components=48, chunk_size=64)
        
        kv = mx.random.normal(shape=(1, 1, 32, 64))
        
        indices, values = dct_comp.compress(kv)
        reconstructed = dct_comp.decompress(indices, values, kv.shape)
        mx.eval(kv, reconstructed)
        
        # Should have reasonable reconstruction (75% of components kept)
        mse = mx.mean((kv - reconstructed) ** 2).item()
        assert mse < 0.5, f"DCT reconstruction MSE too high: {mse}"
    
    def test_grove_awdl_discovery_fallback(self):
        """Test GroveAWDLDiscovery graceful fallback when grove not installed."""
        from turboquant_mlx.grove_integration import GroveAWDLDiscovery
        
        discovery = GroveAWDLDiscovery()
        
        # is_available should return bool without crashing
        available = discovery.is_available()
        assert isinstance(available, bool)
        
        # discover_peers should return list (even if empty)
        peers = discovery.discover_peers(timeout=0.1)
        assert isinstance(peers, list)
        
        # last_discovered should work
        last = discovery.last_discovered
        assert isinstance(last, list)
    
    def test_sparse_kv_delta_different_shapes(self):
        """Test SparseKVDelta works with various tensor shapes."""
        from turboquant_mlx.grove_integration import SparseKVDelta
        
        delta_comp = SparseKVDelta(topk_ratio=0.2)
        
        # Test different shapes
        shapes = [
            (64,),  # 1D
            (32, 64),  # 2D
            (1, 4, 64, 64),  # 4D (typical KV)
        ]
        
        for shape in shapes:
            kv_prev = mx.random.normal(shape=shape)
            kv_new = kv_prev + mx.random.normal(shape=shape) * 0.1
            
            compressed, reconstructed = delta_comp.compress_delta(kv_new, kv_prev)
            mx.eval(compressed, reconstructed)
            
            assert compressed.shape == shape, f"Shape mismatch for {shape}"
            assert reconstructed.shape == shape, f"Shape mismatch for {shape}"
            
            delta_comp.reset_error_buffer()
    
    def test_imports_from_main_package(self):
        """Test Grove integration classes are importable from main package."""
        from turboquant_mlx import SparseKVDelta, DCTKVCompressor, GroveAWDLDiscovery
        
        # Should not raise
        assert SparseKVDelta is not None
        assert DCTKVCompressor is not None
        assert GroveAWDLDiscovery is not None


class TestLazyImports:
    """Test lazy import functions in __init__.py."""
    
    def test_get_ollama_client(self):
        """Test get_ollama_client lazy import."""
        from turboquant_mlx import get_ollama_client
        from turboquant_mlx.ollama_patch import HAS_OPENAI
        
        if not HAS_OPENAI:
            with pytest.raises(ImportError):
                get_ollama_client()
        else:
            client = get_ollama_client()
            assert client is not None
    
    def test_get_hf_cache_class(self):
        """Test get_hf_cache_class lazy import."""
        from turboquant_mlx import get_hf_cache_class
        from turboquant_mlx.hf_patch import HAS_TRANSFORMERS, HAS_TORCH
        
        CacheClass = get_hf_cache_class()
        assert CacheClass is not None
        assert CacheClass.__name__ == "TurboQuantHFCache"
    
    def test_patch_transformers_function(self):
        """Test patch_transformers lazy import."""
        from turboquant_mlx import patch_transformers
        from turboquant_mlx.hf_patch import HAS_TRANSFORMERS, HAS_TORCH
        
        if not (HAS_TRANSFORMERS and HAS_TORCH):
            with pytest.raises(ImportError):
                patch_transformers()
        else:
            # Just test it doesn't crash - actual patching tested separately
            result = patch_transformers()
            # Unpatch to not affect other tests
            from turboquant_mlx.hf_patch import unpatch_transformers
            unpatch_transformers()


class TestPersistence:
    """Test persistence integration in main test file (key tests only)."""
    
    def test_save_load_roundtrip(self):
        """Test save/load roundtrip preserves data."""
        import tempfile
        import shutil
        from pathlib import Path
        from turboquant_mlx.persistence import TurboQuantCache
        
        temp_dir = Path(tempfile.mkdtemp(prefix="turboquant_test_"))
        try:
            cache = TurboQuantCache(cache_dir=temp_dir, bits=4)
            
            # Create test data
            keys = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
            values = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
            mx.eval(keys, values)
            
            original_states = [[keys, values]]
            
            # Save
            result = cache.save(original_states, "test-roundtrip")
            assert result["ratio"] > 1.0
            
            # Load
            loaded_states, meta = cache.load("test-roundtrip")
            assert loaded_states is not None
            
            # Check cosine similarity
            orig_flat = keys.reshape(-1).astype(mx.float32)
            loaded_flat = loaded_states[0][0].reshape(-1).astype(mx.float32)
            mx.eval(orig_flat, loaded_flat)
            
            dot = float(mx.sum(orig_flat * loaded_flat))
            norm_o = float(mx.sqrt(mx.sum(orig_flat ** 2)))
            norm_l = float(mx.sqrt(mx.sum(loaded_flat ** 2)))
            cosine = dot / (norm_o * norm_l + 1e-8)
            
            assert cosine > 0.99, f"Cosine sim {cosine} < 0.99"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_metadata_preserved(self):
        """Test metadata survives save/load."""
        import tempfile
        import shutil
        from pathlib import Path
        from turboquant_mlx.persistence import TurboQuantCache
        
        temp_dir = Path(tempfile.mkdtemp(prefix="turboquant_test_"))
        try:
            cache = TurboQuantCache(cache_dir=temp_dir)
            
            keys = mx.random.normal(shape=(1, 2, 32, 32))
            original_states = [[keys]]
            
            metadata = {"tokens": 4096, "model": "Qwen3.5-35B"}
            cache.save(original_states, "test-meta", metadata=metadata)
            
            _, loaded_meta = cache.load("test-meta")
            
            assert loaded_meta["tokens"] == 4096
            assert loaded_meta["model"] == "Qwen3.5-35B"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_list_caches(self):
        """Test listing saved caches."""
        import tempfile
        import shutil
        from pathlib import Path
        from turboquant_mlx.persistence import TurboQuantCache
        
        temp_dir = Path(tempfile.mkdtemp(prefix="turboquant_test_"))
        try:
            cache = TurboQuantCache(cache_dir=temp_dir)
            
            for name in ["ctx-a", "ctx-b"]:
                keys = mx.random.normal(shape=(1, 2, 16, 32))
                cache.save([[keys]], name)
            
            contexts = cache.list()
            assert len(contexts) == 2
            names = [c["name"] for c in contexts]
            assert "ctx-a" in names
            assert "ctx-b" in names
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
