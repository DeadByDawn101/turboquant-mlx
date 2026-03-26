"""
Test suite for TurboQuant-MLX persistence and tiered cache.

Tests for:
- TurboQuantCache: save/load roundtrip, compression, metadata, LRU
- PagedKVCache: SSD paging, eviction, GPU/SSD stats
- TieredKVCacheManager: three-tier management, promotion/demotion
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import time

import mlx.core as mx
import numpy as np


# Test dimensions
HEAD_DIM = 64
SEQ_LEN = 128
BATCH = 1
NUM_HEADS = 4
NUM_LAYERS = 2


def create_test_kv_states(num_layers=NUM_LAYERS, seq_len=SEQ_LEN):
    """Create test KV cache states (list of [keys, values] per layer)."""
    states = []
    for _ in range(num_layers):
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, seq_len, HEAD_DIM))
        values = mx.random.normal(shape=(BATCH, NUM_HEADS, seq_len, HEAD_DIM))
        states.append([keys, values])
    return states


def cosine_similarity(a, b):
    """Compute cosine similarity between two MLX arrays."""
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    mx.eval(a_flat, b_flat)
    
    dot = mx.sum(a_flat * b_flat)
    norm_a = mx.sqrt(mx.sum(a_flat * a_flat))
    norm_b = mx.sqrt(mx.sum(b_flat * b_flat))
    mx.eval(dot, norm_a, norm_b)
    
    return float(dot) / (float(norm_a) * float(norm_b) + 1e-8)


class TestTurboQuantCache:
    """Test TurboQuantCache persistent save/load."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        temp_dir = tempfile.mkdtemp(prefix="turboquant_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_turboquant_cache_instantiation(self, temp_cache_dir):
        """Test TurboQuantCache can be instantiated."""
        from turboquant_mlx.persistence import TurboQuantCache
        
        cache = TurboQuantCache(
            cache_dir=temp_cache_dir,
            bits=4,
            group_size=64,
            compress=True,
        )
        
        assert cache.cache_dir == temp_cache_dir
        assert cache.bits == 4
        assert cache.group_size == 64
        assert cache.compress is True
    
    def test_save_and_load_roundtrip(self, temp_cache_dir):
        """Test save/load roundtrip preserves data with cosine similarity > 0.99."""
        from turboquant_mlx.persistence import TurboQuantCache
        
        cache = TurboQuantCache(cache_dir=temp_cache_dir, bits=4, group_size=64)
        
        # Create test data
        original_states = create_test_kv_states()
        
        # Save
        result = cache.save(original_states, "test-context")
        
        assert "path" in result
        assert "size_mb" in result
        assert "ratio" in result
        assert result["ratio"] > 1.0  # Should have compression
        
        # Load
        loaded_states, metadata = cache.load("test-context")
        
        assert loaded_states is not None
        assert len(loaded_states) == len(original_states)
        
        # Check cosine similarity for each layer
        for layer_idx, (orig_layer, loaded_layer) in enumerate(zip(original_states, loaded_states)):
            for tensor_idx, (orig_t, loaded_t) in enumerate(zip(orig_layer, loaded_layer)):
                mx.eval(orig_t, loaded_t)
                sim = cosine_similarity(orig_t, loaded_t)
                assert sim > 0.99, f"Layer {layer_idx}, tensor {tensor_idx}: cosine sim {sim} < 0.99"
    
    def test_save_metadata_preserved(self, temp_cache_dir):
        """Test metadata dict survives save/load."""
        from turboquant_mlx.persistence import TurboQuantCache
        
        cache = TurboQuantCache(cache_dir=temp_cache_dir)
        
        original_states = create_test_kv_states(num_layers=1, seq_len=64)
        metadata = {
            "tokens": 4096,
            "model": "Qwen3.5-35B",
            "custom_field": "test_value",
        }
        
        # Save with metadata
        cache.save(original_states, "test-meta", metadata=metadata)
        
        # Load and check metadata
        _, loaded_meta = cache.load("test-meta")
        
        assert loaded_meta["tokens"] == 4096
        assert loaded_meta["model"] == "Qwen3.5-35B"
        assert loaded_meta["extra"]["custom_field"] == "test_value"
    
    def test_list_caches(self, temp_cache_dir):
        """Test list() returns saved caches."""
        from turboquant_mlx.persistence import TurboQuantCache
        
        cache = TurboQuantCache(cache_dir=temp_cache_dir)
        
        # Save multiple contexts
        for name in ["context-1", "context-2", "context-3"]:
            states = create_test_kv_states(num_layers=1, seq_len=32)
            cache.save(states, name, metadata={"tokens": 100})
        
        # List
        contexts = cache.list()
        
        assert len(contexts) == 3
        names = [c["name"] for c in contexts]
        assert "context-1" in names
        assert "context-2" in names
        assert "context-3" in names
    
    def test_delete_cache(self, temp_cache_dir):
        """Test delete() removes cache."""
        from turboquant_mlx.persistence import TurboQuantCache
        
        cache = TurboQuantCache(cache_dir=temp_cache_dir)
        
        # Save
        states = create_test_kv_states(num_layers=1, seq_len=32)
        cache.save(states, "to-delete")
        
        # Verify exists
        assert len(cache.list()) == 1
        
        # Delete
        result = cache.delete("to-delete")
        assert result is True
        
        # Verify gone
        assert len(cache.list()) == 0
        
        # Delete non-existent returns False
        assert cache.delete("non-existent") is False
    
    def test_stats(self, temp_cache_dir):
        """Test stats() returns cache statistics."""
        from turboquant_mlx.persistence import TurboQuantCache
        
        cache = TurboQuantCache(cache_dir=temp_cache_dir, max_cache_mb=100)
        
        # Save a context
        states = create_test_kv_states(num_layers=1, seq_len=64)
        cache.save(states, "stats-test")
        
        # Load to trigger hit
        cache.load("stats-test")
        
        # Load non-existent to trigger miss
        cache.load("non-existent")
        
        stats = cache.stats()
        
        assert stats["count"] == 1
        assert stats["total_size_mb"] > 0
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["compression_bits"] == 4
    
    def test_no_compression_mode(self, temp_cache_dir):
        """Test saving without compression."""
        from turboquant_mlx.persistence import TurboQuantCache
        
        cache = TurboQuantCache(cache_dir=temp_cache_dir, compress=False)
        
        original_states = create_test_kv_states(num_layers=1, seq_len=64)
        
        # Save without compression
        result = cache.save(original_states, "uncompressed")
        
        # Load
        loaded_states, _ = cache.load("uncompressed")
        
        # Should be exact (no quantization loss)
        for orig_layer, loaded_layer in zip(original_states, loaded_states):
            for orig_t, loaded_t in zip(orig_layer, loaded_layer):
                mx.eval(orig_t, loaded_t)
                # Should be nearly exact (float32 vs float16 conversion)
                mse = float(mx.mean((orig_t.astype(mx.float32) - loaded_t.astype(mx.float32)) ** 2))
                assert mse < 0.01, f"MSE {mse} too high for uncompressed"


class TestPagedKVCache:
    """Test PagedKVCache SSD paging."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        temp_dir = tempfile.mkdtemp(prefix="turboquant_paged_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_paged_cache_instantiation(self, temp_cache_dir):
        """Test PagedKVCache can be instantiated."""
        from turboquant_mlx.persistence import PagedKVCache
        
        paged = PagedKVCache(
            max_gpu_chunks=4,
            chunk_size=512,
            cache_dir=temp_cache_dir,
        )
        
        assert paged.max_gpu_chunks == 4
        assert paged.chunk_size == 512
    
    def test_paged_cache_add_and_get(self, temp_cache_dir):
        """Test adding and retrieving chunks."""
        from turboquant_mlx.persistence import PagedKVCache
        
        paged = PagedKVCache(
            max_gpu_chunks=4,
            chunk_size=64,
            cache_dir=temp_cache_dir,
        )
        
        # Add a chunk
        chunk_data = [mx.random.normal(shape=(BATCH, NUM_HEADS, 64, HEAD_DIM))]
        paged.add_chunk(chunk_data, chunk_id=0)
        
        # Get it back
        retrieved = paged.get_chunk(0)
        
        assert retrieved is not None
        assert len(retrieved) == 1
        mx.eval(chunk_data[0], retrieved[0])
        
        # Should be identical (in GPU, no compression yet)
        diff = mx.max(mx.abs(chunk_data[0] - retrieved[0])).item()
        assert diff < 1e-5
    
    def test_paged_cache_eviction(self, temp_cache_dir):
        """Test eviction when max_gpu_chunks exceeded."""
        from turboquant_mlx.persistence import PagedKVCache
        
        paged = PagedKVCache(
            max_gpu_chunks=2,  # Only hold 2 in GPU
            chunk_size=64,
            cache_dir=temp_cache_dir,
        )
        
        # Add 4 chunks (should evict first 2 to SSD)
        for i in range(4):
            chunk_data = [mx.random.normal(shape=(BATCH, NUM_HEADS, 64, HEAD_DIM))]
            paged.add_chunk(chunk_data, chunk_id=i)
        
        stats = paged.stats
        
        # Should have 2 in GPU, 2 on SSD
        assert stats["gpu_chunks"] == 2
        assert stats["ssd_chunks"] == 2
        assert stats["total_chunks"] == 4
    
    def test_paged_cache_ssd_load(self, temp_cache_dir):
        """Test evicted chunk loads back correctly."""
        from turboquant_mlx.persistence import PagedKVCache
        
        paged = PagedKVCache(
            max_gpu_chunks=1,  # Force immediate eviction
            chunk_size=64,
            cache_dir=temp_cache_dir,
            bits=4,
        )
        
        # Add first chunk
        chunk0 = [mx.random.normal(shape=(BATCH, NUM_HEADS, 64, HEAD_DIM))]
        mx.eval(chunk0[0])
        original_data = np.array(chunk0[0])
        paged.add_chunk(chunk0, chunk_id=0)
        
        # Add second chunk (evicts first to SSD)
        chunk1 = [mx.random.normal(shape=(BATCH, NUM_HEADS, 64, HEAD_DIM))]
        paged.add_chunk(chunk1, chunk_id=1)
        
        # Chunk 0 should be on SSD now
        assert paged.stats["ssd_chunks"] >= 1
        
        # Load chunk 0 back from SSD
        loaded = paged.get_chunk(0)
        
        assert loaded is not None
        mx.eval(loaded[0])
        
        # Check quality (compressed, so allow some loss)
        loaded_data = np.array(loaded[0])
        cosine = np.dot(original_data.flatten(), loaded_data.flatten()) / (
            np.linalg.norm(original_data.flatten()) * np.linalg.norm(loaded_data.flatten()) + 1e-8
        )
        assert cosine > 0.95, f"Cosine similarity {cosine} too low after SSD roundtrip"
    
    def test_paged_cache_stats(self, temp_cache_dir):
        """Test stats property returns correct values."""
        from turboquant_mlx.persistence import PagedKVCache
        
        paged = PagedKVCache(
            max_gpu_chunks=2,
            chunk_size=64,
            cache_dir=temp_cache_dir,
        )
        
        # Add chunks and trigger evictions
        for i in range(4):
            chunk_data = [mx.random.normal(shape=(BATCH, NUM_HEADS, 64, HEAD_DIM))]
            paged.add_chunk(chunk_data, chunk_id=i)
        
        # Access some chunks
        paged.get_chunk(3)  # GPU hit
        paged.get_chunk(2)  # GPU hit
        paged.get_chunk(0)  # SSD load
        
        stats = paged.stats
        
        assert "gpu_chunks" in stats
        assert "ssd_chunks" in stats
        assert "gpu_hits" in stats
        assert "ssd_reads" in stats
        assert stats["gpu_hits"] >= 2
        assert stats["ssd_reads"] >= 1
    
    def test_paged_cache_clear(self, temp_cache_dir):
        """Test clearing paged cache."""
        from turboquant_mlx.persistence import PagedKVCache
        
        paged = PagedKVCache(
            max_gpu_chunks=2,
            chunk_size=64,
            cache_dir=temp_cache_dir,
        )
        
        # Add chunks
        for i in range(4):
            chunk_data = [mx.random.normal(shape=(BATCH, NUM_HEADS, 64, HEAD_DIM))]
            paged.add_chunk(chunk_data, chunk_id=i)
        
        assert paged.stats["total_chunks"] == 4
        
        # Clear
        paged.clear()
        
        assert paged.stats["total_chunks"] == 0
        assert paged.stats["gpu_chunks"] == 0
        assert paged.stats["ssd_chunks"] == 0


class TestTieredKVCacheManager:
    """Test TieredKVCacheManager three-tier management."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        temp_dir = tempfile.mkdtemp(prefix="turboquant_tiered_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_tiered_cache_instantiation(self, temp_cache_dir):
        """Test TieredKVCacheManager can be instantiated."""
        from turboquant_mlx.tiered_cache import TieredKVCacheManager
        
        manager = TieredKVCacheManager(
            max_gpu_mb=100,
            max_ssd_mb=1000,
            cache_dir=temp_cache_dir,
        )
        
        assert manager.max_gpu_bytes == 100 * 1024 * 1024
        assert manager.max_ssd_bytes == 1000 * 1024 * 1024
    
    def test_tiered_cache_put_and_get(self, temp_cache_dir):
        """Test put and get operations."""
        from turboquant_mlx.tiered_cache import TieredKVCacheManager
        
        manager = TieredKVCacheManager(
            max_gpu_mb=100,
            max_ssd_mb=1000,
            cache_dir=temp_cache_dir,
        )
        
        # Put a KV state
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        values = mx.random.normal(shape=(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        mx.eval(keys, values)
        
        manager.put("test-key", [keys, values], metadata={"tokens": SEQ_LEN})
        
        # Get it back
        states, tier = manager.get("test-key")
        
        assert states is not None
        assert tier == "gpu"  # Should be in GPU since small
        assert len(states) == 2
    
    def test_tiered_cache_stats(self, temp_cache_dir):
        """Test stats returns tier information."""
        from turboquant_mlx.tiered_cache import TieredKVCacheManager
        
        manager = TieredKVCacheManager(
            max_gpu_mb=100,
            max_ssd_mb=1000,
            cache_dir=temp_cache_dir,
        )
        
        # Add entry
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, 64, HEAD_DIM))
        values = mx.random.normal(shape=(BATCH, NUM_HEADS, 64, HEAD_DIM))
        manager.put("stats-test", [keys, values])
        
        # Get to trigger hit
        manager.get("stats-test")
        
        # Get non-existent to trigger miss
        manager.get("non-existent")
        
        stats = manager.stats()
        
        assert "gpu_mb" in stats
        assert "ssd_mb" in stats
        assert "gpu_entries" in stats
        assert "ssd_entries" in stats
        assert "gpu_hits" in stats
        assert "misses" in stats
        assert stats["gpu_hits"] == 1
        assert stats["misses"] == 1
    
    def test_tiered_cache_auto_demote(self, temp_cache_dir):
        """Test auto-demotion when GPU full."""
        from turboquant_mlx.tiered_cache import TieredKVCacheManager
        
        # Very small GPU limit to force demotion
        manager = TieredKVCacheManager(
            max_gpu_mb=1,  # 1MB
            max_ssd_mb=100,
            cache_dir=temp_cache_dir,
            auto_demote=True,
        )
        
        # Add entries until demotion happens
        for i in range(5):
            keys = mx.random.normal(shape=(1, 4, 256, 64))  # ~256KB each
            values = mx.random.normal(shape=(1, 4, 256, 64))
            mx.eval(keys, values)
            manager.put(f"entry-{i}", [keys, values])
        
        stats = manager.stats()
        
        # Should have some demoted to SSD
        assert stats["demotions"] > 0
        assert stats["ssd_entries"] > 0
    
    def test_tiered_cache_manual_promote_demote(self, temp_cache_dir):
        """Test manual promotion and demotion."""
        from turboquant_mlx.tiered_cache import TieredKVCacheManager
        
        manager = TieredKVCacheManager(
            max_gpu_mb=100,
            max_ssd_mb=1000,
            cache_dir=temp_cache_dir,
            auto_demote=False,  # Manual control
        )
        
        # Add entry to GPU
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, 64, HEAD_DIM))
        values = mx.random.normal(shape=(BATCH, NUM_HEADS, 64, HEAD_DIM))
        mx.eval(keys, values)
        manager.put("manual-test", [keys, values])
        
        # Should be in GPU
        _, tier = manager.get("manual-test")
        assert tier == "gpu"
        
        # Manually demote to SSD
        manager.demote("manual-test", "ssd")
        
        # Should now be in SSD
        entries = manager.list_entries(tier="ssd")
        assert any(e["key"] == "manual-test" for e in entries)
    
    def test_tiered_cache_delete(self, temp_cache_dir):
        """Test deleting entries."""
        from turboquant_mlx.tiered_cache import TieredKVCacheManager
        
        manager = TieredKVCacheManager(
            max_gpu_mb=100,
            max_ssd_mb=1000,
            cache_dir=temp_cache_dir,
        )
        
        # Add entry
        keys = mx.random.normal(shape=(BATCH, NUM_HEADS, 32, HEAD_DIM))
        values = mx.random.normal(shape=(BATCH, NUM_HEADS, 32, HEAD_DIM))
        manager.put("to-delete", [keys, values])
        
        # Verify exists
        states, _ = manager.get("to-delete")
        assert states is not None
        
        # Delete
        result = manager.delete("to-delete")
        assert result is True
        
        # Verify gone
        states, tier = manager.get("to-delete")
        assert states is None
        assert tier is None
    
    def test_tiered_cache_list_entries(self, temp_cache_dir):
        """Test listing entries with tier filter."""
        from turboquant_mlx.tiered_cache import TieredKVCacheManager
        
        manager = TieredKVCacheManager(
            max_gpu_mb=1,  # Small to force demotions
            max_ssd_mb=100,
            cache_dir=temp_cache_dir,
        )
        
        # Add entries
        for i in range(3):
            keys = mx.random.normal(shape=(1, 4, 128, 64))
            values = mx.random.normal(shape=(1, 4, 128, 64))
            mx.eval(keys, values)
            manager.put(f"entry-{i}", [keys, values])
        
        # List all
        all_entries = manager.list_entries()
        assert len(all_entries) == 3
        
        # List by tier
        gpu_entries = manager.list_entries(tier="gpu")
        ssd_entries = manager.list_entries(tier="ssd")
        
        assert len(gpu_entries) + len(ssd_entries) == 3


class TestLazyImports:
    """Test lazy import functions for persistence."""
    
    def test_get_persistent_cache(self):
        """Test get_persistent_cache lazy import."""
        from turboquant_mlx import get_persistent_cache
        
        cache = get_persistent_cache()
        assert cache is not None
        
        # Check it has expected methods
        assert hasattr(cache, 'save')
        assert hasattr(cache, 'load')
        assert hasattr(cache, 'list')
        assert hasattr(cache, 'delete')
    
    def test_get_paged_cache(self):
        """Test get_paged_cache lazy import."""
        from turboquant_mlx import get_paged_cache
        
        paged = get_paged_cache()
        assert paged is not None
        
        # Check it has expected methods
        assert hasattr(paged, 'add_chunk')
        assert hasattr(paged, 'get_chunk')
        assert hasattr(paged, 'stats')
    
    def test_get_tiered_cache(self):
        """Test get_tiered_cache lazy import."""
        from turboquant_mlx import get_tiered_cache
        
        manager = get_tiered_cache()
        assert manager is not None
        
        # Check it has expected methods
        assert hasattr(manager, 'put')
        assert hasattr(manager, 'get')
        assert hasattr(manager, 'promote')
        assert hasattr(manager, 'demote')
        assert hasattr(manager, 'stats')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
