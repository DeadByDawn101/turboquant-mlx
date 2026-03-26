"""
persistence.py — Persistent KV cache save/load with TurboQuant compression.

Combines mac-code's fast save/load (0.0003s) with TurboQuant's 4x compression.

Features:
- Save compressed KV cache to disk in .npz + .meta.json format
- Load in 0.0003s (vs minutes of reprocessing)
- Optional Cloudflare R2 sync for cross-device sharing
- Cache versioning and metadata (model, tokens, timestamp)
- LRU eviction for local cache management
- SSD paging for context beyond GPU memory ("LLM in a Flash")

Based on:
- mac-code's KV persistence: https://github.com/walter-grace/mac-code
- Apple's "LLM in a Flash" research
- TurboQuant compression for 4x smaller cache files

Author: RavenX AI / DeadByDawn101
License: MIT
"""

import json
import time
import gzip
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
from collections import OrderedDict

import mlx.core as mx
import numpy as np


@dataclass
class CacheMetadata:
    """Metadata for a saved KV cache context."""
    name: str
    created: str
    num_layers: int
    total_bytes: int
    compressed_bytes: int
    compression_ratio: float
    bits: int
    group_size: int
    tokens: int = 0
    model: str = ""
    dtype: str = "float16"
    version: str = "1.0"
    extra: Dict[str, Any] = field(default_factory=dict)


class TurboQuantCache:
    """
    Manages persistent TurboQuant-compressed KV caches.
    
    Workflow:
        cache = TurboQuantCache()
        
        # After processing a long document:
        cache.save(kv_states, name="my-project", metadata={"tokens": 4096})
        
        # Next session — load in 0.0003s:
        kv_states, meta = cache.load("my-project")
        
        # Cross-device — sync to R2:
        cache.push("my-project")   # upload
        cache.pull("my-project")   # download on other Mac
    
    Args:
        cache_dir: Directory to store cached contexts (default: ~/.turboquant/kv-cache)
        bits: Quantization bits for compression (2, 3, or 4)
        group_size: Group size for per-group quantization
        compress: Whether to apply TurboQuant compression (vs raw save)
        max_cache_mb: Maximum cache directory size in MB (for LRU eviction)
    """
    
    DEFAULT_CACHE_DIR = Path.home() / ".turboquant" / "kv-cache"
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        bits: int = 4,
        group_size: int = 64,
        compress: bool = True,
        max_cache_mb: int = 10000,  # 10GB default
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.bits = bits
        self.group_size = group_size
        self.compress = compress
        self.max_cache_mb = max_cache_mb
        
        # LRU tracking
        self._access_times: Dict[str, float] = {}
        self._load_access_times()
        
        # Stats
        self._hits = 0
        self._misses = 0
    
    def _load_access_times(self):
        """Load access times from disk for LRU tracking."""
        access_file = self.cache_dir / ".access_times.json"
        if access_file.exists():
            try:
                self._access_times = json.loads(access_file.read_text())
            except (json.JSONDecodeError, IOError):
                self._access_times = {}
    
    def _save_access_times(self):
        """Save access times to disk."""
        access_file = self.cache_dir / ".access_times.json"
        try:
            access_file.write_text(json.dumps(self._access_times))
        except IOError:
            pass
    
    def _quantize_tensor(self, tensor: mx.array) -> Dict[str, Any]:
        """
        Quantize a tensor to N bits with per-group scaling.
        
        Uses asymmetric min-max quantization matching mac-code's approach.
        """
        x = tensor.astype(mx.float32)
        original_shape = x.shape
        
        # Flatten to 2D for group quantization
        x_flat = x.reshape(-1, x.shape[-1])
        rows, cols = x_flat.shape
        
        # Pad columns to multiple of group_size
        pad = (self.group_size - cols % self.group_size) % self.group_size
        if pad > 0:
            x_flat = mx.pad(x_flat, [(0, 0), (0, pad)])
            cols = x_flat.shape[-1]
        
        # Reshape into groups
        n_groups = cols // self.group_size
        x_groups = x_flat.reshape(rows, n_groups, self.group_size)
        
        # Find min/max per group
        g_min = mx.min(x_groups, axis=-1, keepdims=True)
        g_max = mx.max(x_groups, axis=-1, keepdims=True)
        
        # Compute scale and zero point
        max_int = (1 << self.bits) - 1
        scale = (g_max - g_min) / max_int
        scale = mx.where(scale == 0, mx.ones_like(scale), scale)
        zero = g_min
        
        # Quantize
        x_quant = mx.round((x_groups - zero) / scale).astype(mx.uint8)
        x_quant = mx.clip(x_quant, 0, max_int)
        
        return {
            "data": np.array(x_quant),
            "scales": np.array(scale.squeeze(-1).astype(mx.float16)),
            "zeros": np.array(zero.squeeze(-1).astype(mx.float16)),
            "shape": original_shape,
            "dtype": str(tensor.dtype),
            "bits": self.bits,
            "group_size": self.group_size,
        }
    
    def _dequantize_tensor(self, compressed: Dict[str, Any]) -> mx.array:
        """Restore a quantized tensor."""
        data = mx.array(compressed["data"])
        scales = mx.array(compressed["scales"]).astype(mx.float32)
        zeros = mx.array(compressed["zeros"]).astype(mx.float32)
        
        # Dequantize
        x = data.astype(mx.float32) * scales[..., None] + zeros[..., None]
        
        # Reshape back
        x = x.reshape(-1, x.shape[-2] * x.shape[-1])
        
        # Trim padding and restore shape
        target_size = 1
        for s in compressed["shape"]:
            target_size *= s
        x = x.reshape(-1)[:target_size]
        x = x.reshape(compressed["shape"])
        
        # Convert back to original dtype
        dtype_str = compressed.get("dtype", "float16")
        if "bfloat16" in dtype_str:
            x = x.astype(mx.bfloat16)
        elif "float16" in dtype_str:
            x = x.astype(mx.float16)
        
        return x
    
    def save(
        self,
        kv_states: Union[List[mx.array], List[List[mx.array]]],
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save KV cache with TurboQuant compression.
        
        Args:
            kv_states: List of KV cache tensors (per layer), or list of [keys, values] pairs
            name: Name for the saved context
            metadata: Optional metadata dict (model, tokens, etc.)
            
        Returns:
            dict: {"path": ..., "size_mb": ..., "ratio": ..., "save_time_s": ...}
        """
        t0 = time.time()
        
        cache_path = self.cache_dir / name
        cache_path.mkdir(parents=True, exist_ok=True)
        
        arrays = {}
        meta_layers = []
        original_bytes = 0
        compressed_bytes = 0
        
        # Process each layer's state
        for layer_idx, layer_state in enumerate(kv_states):
            layer_meta = []
            
            # Handle both list-of-tensors and nested list formats
            if isinstance(layer_state, list):
                tensors = layer_state
            elif hasattr(layer_state, '__iter__') and not hasattr(layer_state, 'shape'):
                tensors = list(layer_state)
            else:
                tensors = [layer_state]
            
            for tensor_idx, tensor in enumerate(tensors):
                if tensor is None or not hasattr(tensor, 'shape'):
                    continue
                    
                prefix = f"l{layer_idx}_t{tensor_idx}"
                original_bytes += tensor.nbytes
                
                if self.compress:
                    # Apply TurboQuant compression
                    comp = self._quantize_tensor(tensor)
                    arrays[f"{prefix}_data"] = comp["data"]
                    arrays[f"{prefix}_scales"] = comp["scales"]
                    arrays[f"{prefix}_zeros"] = comp["zeros"]
                    
                    compressed_bytes += (
                        comp["data"].nbytes + 
                        comp["scales"].nbytes + 
                        comp["zeros"].nbytes
                    )
                    
                    layer_meta.append({
                        "shape": list(comp["shape"]),
                        "dtype": comp["dtype"],
                        "bits": comp["bits"],
                        "group_size": comp["group_size"],
                        "compressed": True,
                    })
                else:
                    # Save raw tensor
                    arrays[f"{prefix}"] = np.array(tensor)
                    compressed_bytes += tensor.nbytes
                    
                    layer_meta.append({
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "compressed": False,
                    })
            
            meta_layers.append(layer_meta)
        
        # Force evaluation before saving
        mx.eval(*[mx.array(v) for v in arrays.values()])
        
        # Save arrays
        npz_path = cache_path / "kv_cache.npz"
        np.savez_compressed(str(npz_path), **arrays)
        
        # Calculate actual file size
        actual_size = npz_path.stat().st_size
        
        # Build metadata
        cache_meta = CacheMetadata(
            name=name,
            created=datetime.now().isoformat(),
            num_layers=len(kv_states),
            total_bytes=original_bytes,
            compressed_bytes=actual_size,
            compression_ratio=original_bytes / actual_size if actual_size > 0 else 1.0,
            bits=self.bits if self.compress else 16,
            group_size=self.group_size if self.compress else 0,
            tokens=metadata.get("tokens", 0) if metadata else 0,
            model=metadata.get("model", "") if metadata else "",
            dtype=str(kv_states[0][0].dtype) if kv_states and isinstance(kv_states[0], list) else "unknown",
        )
        
        if metadata:
            cache_meta.extra = {k: v for k, v in metadata.items() if k not in ("tokens", "model")}
        
        # Save metadata
        full_meta = {
            "layers": meta_layers,
            **vars(cache_meta),
        }
        meta_path = cache_path / "metadata.json"
        meta_path.write_text(json.dumps(full_meta, indent=2))
        
        save_time = time.time() - t0
        
        # Update access time
        self._access_times[name] = time.time()
        self._save_access_times()
        
        # Check if we need to evict old caches
        self._maybe_evict()
        
        return {
            "path": str(cache_path),
            "size_mb": actual_size / (1024 * 1024),
            "ratio": cache_meta.compression_ratio,
            "save_time_s": save_time,
            "original_mb": original_bytes / (1024 * 1024),
        }
    
    def load(self, name: str) -> Tuple[List[List[mx.array]], Dict[str, Any]]:
        """
        Load compressed KV cache.
        
        Args:
            name: Name of the saved context
            
        Returns:
            tuple: (kv_states, metadata)
                - kv_states: List of layer states, each layer is [keys, values, ...]
                - metadata: Dict with cache metadata
        """
        t0 = time.time()
        
        cache_path = self.cache_dir / name
        
        if not cache_path.exists():
            self._misses += 1
            return None, {"error": f"Cache not found: {name}"}
        
        # Load metadata
        meta_path = cache_path / "metadata.json"
        if not meta_path.exists():
            self._misses += 1
            return None, {"error": f"Metadata not found for: {name}"}
        
        metadata = json.loads(meta_path.read_text())
        
        # Load arrays
        npz_path = cache_path / "kv_cache.npz"
        data = np.load(str(npz_path), allow_pickle=True)
        
        # Reconstruct KV states
        kv_states = []
        for layer_idx, layer_meta in enumerate(metadata["layers"]):
            layer_tensors = []
            for tensor_idx, tensor_meta in enumerate(layer_meta):
                prefix = f"l{layer_idx}_t{tensor_idx}"
                
                if tensor_meta.get("compressed", False):
                    # Decompress
                    comp = {
                        "data": data[f"{prefix}_data"],
                        "scales": data[f"{prefix}_scales"],
                        "zeros": data[f"{prefix}_zeros"],
                        "shape": tuple(tensor_meta["shape"]),
                        "dtype": tensor_meta["dtype"],
                        "bits": tensor_meta["bits"],
                        "group_size": tensor_meta["group_size"],
                    }
                    tensor = self._dequantize_tensor(comp)
                else:
                    # Load raw
                    tensor = mx.array(data[prefix])
                
                layer_tensors.append(tensor)
            
            kv_states.append(layer_tensors)
        
        load_time = time.time() - t0
        
        # Update access time
        self._access_times[name] = time.time()
        self._save_access_times()
        self._hits += 1
        
        # Add load time to metadata
        metadata["load_time_s"] = load_time
        
        return kv_states, metadata
    
    def list(self) -> List[Dict[str, Any]]:
        """List all cached contexts with size and metadata."""
        contexts = []
        
        for path in self.cache_dir.iterdir():
            if not path.is_dir() or path.name.startswith('.'):
                continue
            
            meta_path = path / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    
                    # Add file sizes
                    npz_path = path / "kv_cache.npz"
                    meta["disk_size_mb"] = npz_path.stat().st_size / (1024 * 1024) if npz_path.exists() else 0
                    
                    # Add last access time
                    meta["last_accessed"] = self._access_times.get(path.name, 0)
                    
                    contexts.append(meta)
                except (json.JSONDecodeError, IOError):
                    continue
        
        return sorted(contexts, key=lambda x: x.get("created", ""), reverse=True)
    
    def delete(self, name: str) -> bool:
        """Delete a cached context."""
        cache_path = self.cache_dir / name
        if cache_path.exists():
            shutil.rmtree(cache_path)
            self._access_times.pop(name, None)
            self._save_access_times()
            return True
        return False
    
    def push(self, name: str, r2_config: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Upload to Cloudflare R2 (if configured).
        
        Args:
            name: Name of the cache to upload
            r2_config: Optional R2 config dict with endpoint, access_key, secret_key, bucket
            
        Returns:
            dict: Upload stats or error
        """
        cache_path = self.cache_dir / name
        
        if not cache_path.exists():
            return {"error": f"Cache not found: {name}"}
        
        # Get R2 client
        client, bucket = self._get_r2_client(r2_config)
        if not client:
            return {"error": "R2 not configured. Set R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET env vars."}
        
        t0 = time.time()
        
        # Compress the npz file with gzip for smaller transfer
        npz_path = cache_path / "kv_cache.npz"
        gz_path = cache_path / "kv_cache.npz.gz"
        
        with open(npz_path, "rb") as f_in:
            with gzip.open(str(gz_path), "wb", compresslevel=6) as f_out:
                f_out.write(f_in.read())
        
        # Upload compressed file
        key = f"turboquant/{name}/kv_cache.npz.gz"
        client.upload_file(str(gz_path), bucket, key)
        
        # Upload metadata
        meta_path = cache_path / "metadata.json"
        client.upload_file(str(meta_path), bucket, f"turboquant/{name}/metadata.json")
        
        upload_time = time.time() - t0
        size_mb = gz_path.stat().st_size / (1024 * 1024)
        
        # Clean up temp gzip
        gz_path.unlink()
        
        return {
            "name": name,
            "compressed_mb": size_mb,
            "upload_time_s": upload_time,
            "speed_mbps": size_mb / upload_time if upload_time > 0 else 0,
        }
    
    def pull(self, name: str, r2_config: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Download from Cloudflare R2.
        
        Args:
            name: Name of the cache to download
            r2_config: Optional R2 config dict
            
        Returns:
            dict: Download stats or error
        """
        client, bucket = self._get_r2_client(r2_config)
        if not client:
            return {"error": "R2 not configured"}
        
        cache_path = self.cache_dir / name
        cache_path.mkdir(parents=True, exist_ok=True)
        
        t0 = time.time()
        
        # Download compressed file
        gz_path = cache_path / "kv_cache.npz.gz"
        npz_path = cache_path / "kv_cache.npz"
        
        try:
            client.download_file(bucket, f"turboquant/{name}/kv_cache.npz.gz", str(gz_path))
            client.download_file(bucket, f"turboquant/{name}/metadata.json", str(cache_path / "metadata.json"))
        except Exception as e:
            return {"error": f"Download failed: {e}"}
        
        download_time = time.time() - t0
        
        # Decompress
        with gzip.open(str(gz_path), "rb") as f_in:
            with open(npz_path, "wb") as f_out:
                f_out.write(f_in.read())
        
        total_time = time.time() - t0
        size_mb = gz_path.stat().st_size / (1024 * 1024)
        
        # Clean up
        gz_path.unlink()
        
        return {
            "name": name,
            "compressed_mb": size_mb,
            "download_time_s": download_time,
            "total_time_s": total_time,
            "speed_mbps": size_mb / download_time if download_time > 0 else 0,
        }
    
    def stats(self) -> Dict[str, Any]:
        """Return cache stats: total size, count, hit rate."""
        contexts = self.list()
        total_size_mb = sum(c.get("disk_size_mb", 0) for c in contexts)
        
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_dir": str(self.cache_dir),
            "count": len(contexts),
            "total_size_mb": total_size_mb,
            "max_size_mb": self.max_cache_mb,
            "utilization_pct": (total_size_mb / self.max_cache_mb * 100) if self.max_cache_mb > 0 else 0,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "compression_bits": self.bits,
            "group_size": self.group_size,
        }
    
    def _get_r2_client(self, r2_config: Optional[Dict[str, str]] = None):
        """Get boto3 S3 client configured for Cloudflare R2."""
        import os
        
        try:
            import boto3
        except ImportError:
            return None, None
        
        config = r2_config or {}
        
        # Try config file
        config_path = Path.home() / ".turboquant" / "r2-config.json"
        if config_path.exists() and not r2_config:
            try:
                config = json.loads(config_path.read_text())
            except (json.JSONDecodeError, IOError):
                pass
        
        endpoint = config.get("endpoint") or os.environ.get("R2_ENDPOINT")
        access_key = config.get("access_key") or os.environ.get("R2_ACCESS_KEY")
        secret_key = config.get("secret_key") or os.environ.get("R2_SECRET_KEY")
        bucket = config.get("bucket") or os.environ.get("R2_BUCKET", "turboquant-kv")
        
        if not all([endpoint, access_key, secret_key]):
            return None, None
        
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        return client, bucket
    
    def _maybe_evict(self):
        """Evict oldest caches if over size limit (LRU)."""
        contexts = self.list()
        total_size_mb = sum(c.get("disk_size_mb", 0) for c in contexts)
        
        if total_size_mb <= self.max_cache_mb:
            return
        
        # Sort by last accessed (oldest first)
        sorted_contexts = sorted(
            contexts,
            key=lambda x: self._access_times.get(x.get("name", ""), 0)
        )
        
        for ctx in sorted_contexts:
            if total_size_mb <= self.max_cache_mb * 0.8:  # Evict to 80% capacity
                break
            
            name = ctx.get("name")
            if name:
                size = ctx.get("disk_size_mb", 0)
                self.delete(name)
                total_size_mb -= size


@dataclass
class PagedChunk:
    """Metadata for a KV cache chunk (GPU or SSD)."""
    chunk_id: int
    start_token: int
    end_token: int
    in_gpu: bool
    ssd_path: Optional[str] = None
    size_mb: float = 0.0
    last_accessed: float = 0.0


class PagedKVCache:
    """
    SSD-paged KV cache for context beyond GPU memory ("LLM in a Flash").
    
    Splits the KV cache into chunks. GPU holds the most recent N chunks.
    Older chunks are evicted to SSD. On access, swapped back in.
    
    This enables processing documents larger than GPU memory while keeping
    full attention history available.
    
    Based on Apple's "LLM in a Flash" research + mac-code's implementation.
    
    Args:
        max_gpu_chunks: Maximum number of chunks to keep in GPU memory
        chunk_size: Number of tokens per chunk
        cache_dir: Directory for SSD-paged chunks
        bits: Quantization bits for SSD compression
        group_size: Group size for quantization
        
    Example:
        paged = PagedKVCache(max_gpu_chunks=4, chunk_size=512)
        
        for chunk_id, kv_chunk in enumerate(kv_chunks):
            paged.add_chunk(kv_chunk, chunk_id)
        
        print(paged.stats)
        # {"gpu_chunks": 4, "ssd_chunks": 12, "gpu_hits": 89, "ssd_reads": 11}
    """
    
    def __init__(
        self,
        max_gpu_chunks: int = 4,
        chunk_size: int = 512,
        cache_dir: Optional[Path] = None,
        bits: int = 4,
        group_size: int = 64,
    ):
        self.max_gpu_chunks = max_gpu_chunks
        self.chunk_size = chunk_size
        self.cache_dir = Path(cache_dir) if cache_dir else (Path.home() / ".turboquant" / "paged-cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.bits = bits
        self.group_size = group_size
        
        # Create a TurboQuantCache instance for compression
        self._compressor = TurboQuantCache(
            cache_dir=self.cache_dir / "compressed",
            bits=bits,
            group_size=group_size,
            compress=True,
        )
        
        # GPU-resident chunks: chunk_id -> list of tensors
        self._gpu_chunks: OrderedDict[int, List[mx.array]] = OrderedDict()
        
        # All chunk metadata
        self._chunks: Dict[int, PagedChunk] = {}
        
        # Stats
        self._gpu_hits = 0
        self._ssd_reads = 0
        self._total_tokens = 0
    
    def add_chunk(self, kv_states: List[mx.array], chunk_id: int) -> None:
        """
        Add a new KV chunk. Evicts oldest to SSD if at capacity.
        
        Args:
            kv_states: List of KV tensors for this chunk
            chunk_id: Unique identifier for this chunk
        """
        # Calculate token range
        start_token = chunk_id * self.chunk_size
        end_token = start_token + self.chunk_size
        
        # Calculate size
        total_bytes = sum(t.nbytes for t in kv_states if hasattr(t, 'nbytes'))
        size_mb = total_bytes / (1024 * 1024)
        
        # Create chunk metadata
        chunk = PagedChunk(
            chunk_id=chunk_id,
            start_token=start_token,
            end_token=end_token,
            in_gpu=True,
            size_mb=size_mb,
            last_accessed=time.time(),
        )
        
        self._chunks[chunk_id] = chunk
        self._gpu_chunks[chunk_id] = kv_states
        self._total_tokens = max(self._total_tokens, end_token)
        
        # Evict if over capacity
        while len(self._gpu_chunks) > self.max_gpu_chunks:
            # Get oldest chunk (first in OrderedDict)
            oldest_id = next(iter(self._gpu_chunks))
            self.evict_to_ssd(oldest_id)
    
    def get_chunk(self, chunk_id: int) -> Optional[List[mx.array]]:
        """
        Get a KV chunk, loading from SSD if needed.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            List of KV tensors, or None if chunk doesn't exist
        """
        if chunk_id not in self._chunks:
            return None
        
        chunk = self._chunks[chunk_id]
        chunk.last_accessed = time.time()
        
        if chunk.in_gpu:
            # Move to end of OrderedDict (most recently used)
            self._gpu_chunks.move_to_end(chunk_id)
            self._gpu_hits += 1
            return self._gpu_chunks[chunk_id]
        else:
            # Load from SSD
            return self.load_from_ssd(chunk_id)
    
    def evict_to_ssd(self, chunk_id: int) -> None:
        """
        Compress and save a chunk to SSD.
        
        Args:
            chunk_id: ID of the chunk to evict
        """
        if chunk_id not in self._gpu_chunks:
            return
        
        chunk = self._chunks[chunk_id]
        kv_states = self._gpu_chunks[chunk_id]
        
        # Save compressed to SSD
        chunk_name = f"chunk_{chunk_id}"
        result = self._compressor.save(
            [[t] for t in kv_states],  # Wrap each tensor as a "layer"
            chunk_name,
            metadata={"chunk_id": chunk_id, "start_token": chunk.start_token, "end_token": chunk.end_token}
        )
        
        # Update chunk metadata
        chunk.in_gpu = False
        chunk.ssd_path = result["path"]
        chunk.size_mb = result["size_mb"]
        
        # Remove from GPU
        del self._gpu_chunks[chunk_id]
    
    def load_from_ssd(self, chunk_id: int) -> Optional[List[mx.array]]:
        """
        Load and decompress a chunk from SSD.
        
        Args:
            chunk_id: ID of the chunk to load
            
        Returns:
            List of KV tensors
        """
        chunk = self._chunks.get(chunk_id)
        if not chunk or chunk.in_gpu:
            return self._gpu_chunks.get(chunk_id)
        
        # Load from SSD
        chunk_name = f"chunk_{chunk_id}"
        kv_states, _ = self._compressor.load(chunk_name)
        
        if kv_states is None:
            return None
        
        # Unwrap the layer format
        tensors = [layer[0] for layer in kv_states if layer]
        
        self._ssd_reads += 1
        
        # Add back to GPU (may trigger another eviction)
        chunk.in_gpu = True
        self._gpu_chunks[chunk_id] = tensors
        
        # Evict if over capacity
        while len(self._gpu_chunks) > self.max_gpu_chunks:
            oldest_id = next(iter(self._gpu_chunks))
            if oldest_id != chunk_id:  # Don't evict what we just loaded
                self.evict_to_ssd(oldest_id)
        
        return tensors
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Return {gpu_chunks, ssd_chunks, ssd_reads, gpu_hits}."""
        gpu_chunks = sum(1 for c in self._chunks.values() if c.in_gpu)
        ssd_chunks = sum(1 for c in self._chunks.values() if not c.in_gpu)
        
        return {
            "gpu_chunks": gpu_chunks,
            "ssd_chunks": ssd_chunks,
            "total_chunks": len(self._chunks),
            "gpu_hits": self._gpu_hits,
            "ssd_reads": self._ssd_reads,
            "total_tokens": self._total_tokens,
            "gpu_mb": sum(c.size_mb for c in self._chunks.values() if c.in_gpu),
            "ssd_mb": sum(c.size_mb for c in self._chunks.values() if not c.in_gpu),
        }
    
    def clear(self) -> None:
        """Clear all chunks from GPU and SSD."""
        self._gpu_chunks.clear()
        self._chunks.clear()
        self._gpu_hits = 0
        self._ssd_reads = 0
        self._total_tokens = 0
        
        # Clean up SSD directory
        for path in (self.cache_dir / "compressed").iterdir():
            if path.is_dir() and path.name.startswith("chunk_"):
                shutil.rmtree(path)
