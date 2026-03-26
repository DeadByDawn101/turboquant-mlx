"""
tiered_cache.py — Three-tier KV cache: GPU (hot) → SSD (warm) → R2 (cold)

Tier 1 (GPU): Recent KV cache in unified memory — instant access
Tier 2 (SSD): Compressed KV checkpoints on local SSD — 0.0003s access
Tier 3 (R2): Cloudflare R2 object storage — 1.5s access, cross-device

The manager automatically promotes/demotes based on access patterns.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  TIER 1: GPU Unified Memory (HOT)                       │
    │  • Instant access (0ms)                                 │
    │  • Recent tokens + attention sinks                      │
    │  • Size: max_gpu_mb (default 2GB)                       │
    ├─────────────────────────────────────────────────────────┤
    │  TIER 2: Local SSD (WARM)                               │
    │  • Fast access (~0.3ms for load)                        │
    │  • TurboQuant 4x compressed                             │
    │  • Size: max_ssd_mb (default 50GB)                      │
    ├─────────────────────────────────────────────────────────┤
    │  TIER 3: Cloudflare R2 (COLD)                           │
    │  • Network access (~1.5s)                               │
    │  • Cross-device persistence                             │
    │  • Free tier: 10GB, no egress fees                      │
    └─────────────────────────────────────────────────────────┘

Based on:
- mac-code's tiered_cache.py: https://github.com/walter-grace/mac-code
- LMCache: Multi-tier KV caching (GPU → CPU → S3)
- Apple's unified memory architecture

Author: RavenX AI / DeadByDawn101
License: MIT
"""

import json
import time
import gzip
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
from collections import OrderedDict
import threading

import mlx.core as mx
import numpy as np

from .persistence import TurboQuantCache


@dataclass
class CacheEntry:
    """Metadata for a cached KV state across tiers."""
    key: str
    tier: str  # "gpu", "ssd", "r2"
    size_bytes: int
    created: float
    last_accessed: float
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Tier-specific data
    gpu_data: Optional[List[mx.array]] = None
    ssd_path: Optional[str] = None
    r2_key: Optional[str] = None


class TieredKVCacheManager:
    """
    Three-tier KV cache manager: GPU → SSD → R2.
    
    Automatically promotes/demotes entries based on access patterns.
    
    Args:
        max_gpu_mb: Maximum GPU tier size in MB (default: 2000)
        max_ssd_mb: Maximum SSD tier size in MB (default: 50000)
        r2_config: Optional R2 configuration dict
        cache_dir: Base directory for SSD cache
        bits: Quantization bits for compression
        group_size: Group size for quantization
        auto_promote: Automatically promote on access
        auto_demote: Automatically demote on memory pressure
        
    Example:
        manager = TieredKVCacheManager(max_gpu_mb=2000, max_ssd_mb=50000)
        
        # Store KV state
        manager.put("my-context", kv_states, metadata={"tokens": 4096})
        
        # Retrieve (auto-promotes if from lower tier)
        states, tier = manager.get("my-context")
        print(f"Retrieved from {tier}")  # "gpu", "ssd", or "r2"
        
        # Check stats
        print(manager.stats())
    """
    
    def __init__(
        self,
        max_gpu_mb: int = 2000,
        max_ssd_mb: int = 50000,
        r2_config: Optional[Dict[str, str]] = None,
        cache_dir: Optional[Path] = None,
        bits: int = 4,
        group_size: int = 64,
        auto_promote: bool = True,
        auto_demote: bool = True,
    ):
        self.max_gpu_bytes = max_gpu_mb * 1024 * 1024
        self.max_ssd_bytes = max_ssd_mb * 1024 * 1024
        self.r2_config = r2_config
        
        self.cache_dir = Path(cache_dir) if cache_dir else (Path.home() / ".turboquant" / "tiered-cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.bits = bits
        self.group_size = group_size
        self.auto_promote = auto_promote
        self.auto_demote = auto_demote
        
        # SSD tier uses TurboQuantCache for compression
        self._ssd_cache = TurboQuantCache(
            cache_dir=self.cache_dir / "ssd",
            bits=bits,
            group_size=group_size,
            compress=True,
        )
        
        # Entry tracking (all tiers)
        self._entries: Dict[str, CacheEntry] = {}
        
        # GPU tier: LRU ordered dict
        self._gpu_data: OrderedDict[str, List[mx.array]] = OrderedDict()
        
        # Current tier sizes
        self._gpu_bytes = 0
        self._ssd_bytes = 0
        self._r2_bytes = 0
        
        # Stats
        self._stats = {
            "gpu_hits": 0,
            "ssd_hits": 0,
            "r2_hits": 0,
            "misses": 0,
            "promotions": 0,
            "demotions": 0,
        }
        
        # Thread lock for concurrent access
        self._lock = threading.Lock()
        
        # Load existing SSD entries
        self._scan_ssd_tier()
    
    def _scan_ssd_tier(self):
        """Scan SSD directory and populate entry metadata."""
        ssd_dir = self.cache_dir / "ssd"
        if not ssd_dir.exists():
            return
        
        for path in ssd_dir.iterdir():
            if not path.is_dir() or path.name.startswith('.'):
                continue
            
            meta_path = path / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    key = meta.get("name", path.name)
                    
                    npz_path = path / "kv_cache.npz"
                    size_bytes = npz_path.stat().st_size if npz_path.exists() else 0
                    
                    entry = CacheEntry(
                        key=key,
                        tier="ssd",
                        size_bytes=size_bytes,
                        created=time.mktime(datetime.fromisoformat(meta.get("created", datetime.now().isoformat())).timetuple()),
                        last_accessed=time.time(),
                        metadata=meta,
                        ssd_path=str(path),
                    )
                    
                    self._entries[key] = entry
                    self._ssd_bytes += size_bytes
                except (json.JSONDecodeError, IOError, ValueError):
                    continue
    
    def get(self, key: str) -> Tuple[Optional[List[mx.array]], str]:
        """
        Get KV state. Returns (states, tier_hit) where tier_hit is 'gpu'|'ssd'|'r2'|None.
        
        Auto-promotes to higher tier if auto_promote is enabled.
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            tuple: (kv_states or None, tier hit string or None)
        """
        with self._lock:
            if key not in self._entries:
                self._stats["misses"] += 1
                return None, None
            
            entry = self._entries[key]
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            if entry.tier == "gpu":
                self._stats["gpu_hits"] += 1
                # Move to end of LRU
                self._gpu_data.move_to_end(key)
                return self._gpu_data[key], "gpu"
            
            elif entry.tier == "ssd":
                self._stats["ssd_hits"] += 1
                # Load from SSD
                states, meta = self._ssd_cache.load(key)
                
                if states is None:
                    return None, None
                
                # Flatten nested structure
                flat_states = []
                for layer in states:
                    if isinstance(layer, list):
                        flat_states.extend(layer)
                    else:
                        flat_states.append(layer)
                
                # Auto-promote to GPU if enabled and space available
                if self.auto_promote:
                    self._promote_to_gpu(key, flat_states, entry)
                
                return flat_states, "ssd"
            
            elif entry.tier == "r2":
                self._stats["r2_hits"] += 1
                # Pull from R2
                result = self._ssd_cache.pull(key, self.r2_config)
                
                if "error" in result:
                    return None, None
                
                # Now load from SSD
                states, meta = self._ssd_cache.load(key)
                
                if states is None:
                    return None, None
                
                # Flatten
                flat_states = []
                for layer in states:
                    if isinstance(layer, list):
                        flat_states.extend(layer)
                    else:
                        flat_states.append(layer)
                
                # Update entry to SSD tier
                entry.tier = "ssd"
                entry.ssd_path = str(self.cache_dir / "ssd" / key)
                
                # Auto-promote to GPU if enabled
                if self.auto_promote:
                    self._promote_to_gpu(key, flat_states, entry)
                
                return flat_states, "r2"
        
        return None, None
    
    def put(
        self,
        key: str,
        states: List[mx.array],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store KV state. Automatically tiers based on size/recency.
        
        Small/recent entries go to GPU, larger ones to SSD, oldest to R2.
        
        Args:
            key: Cache key
            states: List of KV tensors
            metadata: Optional metadata dict
        """
        with self._lock:
            # Calculate size
            size_bytes = sum(t.nbytes for t in states if hasattr(t, 'nbytes'))
            
            # Create entry
            entry = CacheEntry(
                key=key,
                tier="gpu",  # Start in GPU
                size_bytes=size_bytes,
                created=time.time(),
                last_accessed=time.time(),
                metadata=metadata or {},
                gpu_data=states,
            )
            
            # Check if we need to demote existing entries
            if self.auto_demote:
                self._ensure_gpu_space(size_bytes)
            
            # Add to GPU tier
            self._entries[key] = entry
            self._gpu_data[key] = states
            self._gpu_bytes += size_bytes
            
            # If still over capacity, demote this entry immediately
            if self._gpu_bytes > self.max_gpu_bytes:
                self._demote_from_gpu(key)
    
    def promote(self, key: str, to_tier: str) -> None:
        """
        Manually promote a cache entry to a higher tier.
        
        Args:
            key: Cache key to promote
            to_tier: Target tier ("gpu" or "ssd")
        """
        with self._lock:
            if key not in self._entries:
                return
            
            entry = self._entries[key]
            
            if to_tier == "gpu" and entry.tier != "gpu":
                # Load and promote to GPU
                if entry.tier == "r2":
                    # First pull to SSD
                    self._ssd_cache.pull(key, self.r2_config)
                    entry.tier = "ssd"
                
                if entry.tier == "ssd":
                    # Load from SSD
                    states, _ = self._ssd_cache.load(key)
                    if states:
                        flat_states = []
                        for layer in states:
                            if isinstance(layer, list):
                                flat_states.extend(layer)
                            else:
                                flat_states.append(layer)
                        self._promote_to_gpu(key, flat_states, entry)
            
            elif to_tier == "ssd" and entry.tier == "r2":
                # Pull from R2 to SSD
                self._ssd_cache.pull(key, self.r2_config)
                entry.tier = "ssd"
                entry.ssd_path = str(self.cache_dir / "ssd" / key)
                self._stats["promotions"] += 1
    
    def demote(self, key: str, to_tier: str) -> None:
        """
        Manually demote a cache entry to a lower tier.
        
        Args:
            key: Cache key to demote
            to_tier: Target tier ("ssd" or "r2")
        """
        with self._lock:
            if key not in self._entries:
                return
            
            entry = self._entries[key]
            
            if to_tier == "ssd" and entry.tier == "gpu":
                self._demote_from_gpu(key)
            
            elif to_tier == "r2":
                # Ensure in SSD first
                if entry.tier == "gpu":
                    self._demote_from_gpu(key)
                
                if entry.tier == "ssd":
                    # Push to R2
                    result = self._ssd_cache.push(key, self.r2_config)
                    if "error" not in result:
                        # Update entry
                        entry.tier = "r2"
                        entry.r2_key = f"turboquant/{key}"
                        self._r2_bytes += entry.size_bytes
                        
                        # Optionally remove from SSD to save space
                        # self._ssd_cache.delete(key)
                        # self._ssd_bytes -= entry.size_bytes
                        
                        self._stats["demotions"] += 1
    
    def stats(self) -> Dict[str, Any]:
        """Return tier utilization and hit rates."""
        total_hits = self._stats["gpu_hits"] + self._stats["ssd_hits"] + self._stats["r2_hits"]
        total_requests = total_hits + self._stats["misses"]
        
        gpu_entries = sum(1 for e in self._entries.values() if e.tier == "gpu")
        ssd_entries = sum(1 for e in self._entries.values() if e.tier == "ssd")
        r2_entries = sum(1 for e in self._entries.values() if e.tier == "r2")
        
        return {
            # Sizes
            "gpu_mb": self._gpu_bytes / (1024 * 1024),
            "gpu_max_mb": self.max_gpu_bytes / (1024 * 1024),
            "gpu_utilization_pct": (self._gpu_bytes / self.max_gpu_bytes * 100) if self.max_gpu_bytes > 0 else 0,
            
            "ssd_mb": self._ssd_bytes / (1024 * 1024),
            "ssd_max_mb": self.max_ssd_bytes / (1024 * 1024),
            "ssd_utilization_pct": (self._ssd_bytes / self.max_ssd_bytes * 100) if self.max_ssd_bytes > 0 else 0,
            
            "r2_mb": self._r2_bytes / (1024 * 1024),
            
            # Entry counts
            "gpu_entries": gpu_entries,
            "ssd_entries": ssd_entries,
            "r2_entries": r2_entries,
            "total_entries": len(self._entries),
            
            # Hit rates
            "gpu_hits": self._stats["gpu_hits"],
            "ssd_hits": self._stats["ssd_hits"],
            "r2_hits": self._stats["r2_hits"],
            "misses": self._stats["misses"],
            "hit_rate": total_hits / total_requests if total_requests > 0 else 0.0,
            
            # Operations
            "promotions": self._stats["promotions"],
            "demotions": self._stats["demotions"],
            
            # Config
            "compression_bits": self.bits,
            "group_size": self.group_size,
            "auto_promote": self.auto_promote,
            "auto_demote": self.auto_demote,
        }
    
    def _promote_to_gpu(self, key: str, states: List[mx.array], entry: CacheEntry) -> None:
        """Promote an entry from SSD/R2 to GPU tier."""
        size_bytes = sum(t.nbytes for t in states if hasattr(t, 'nbytes'))
        
        # Ensure space
        self._ensure_gpu_space(size_bytes)
        
        # Add to GPU
        self._gpu_data[key] = states
        self._gpu_bytes += size_bytes
        
        # Update entry
        if entry.tier == "ssd":
            self._ssd_bytes -= entry.size_bytes
        
        entry.tier = "gpu"
        entry.size_bytes = size_bytes
        entry.gpu_data = states
        
        self._stats["promotions"] += 1
    
    def _demote_from_gpu(self, key: str) -> None:
        """Demote an entry from GPU to SSD tier."""
        if key not in self._gpu_data:
            return
        
        entry = self._entries.get(key)
        if not entry:
            return
        
        states = self._gpu_data[key]
        
        # Save to SSD
        result = self._ssd_cache.save(
            [[t] for t in states],  # Wrap as layers
            key,
            metadata=entry.metadata,
        )
        
        # Update tracking
        self._gpu_bytes -= entry.size_bytes
        self._ssd_bytes += int(result["size_mb"] * 1024 * 1024)
        
        entry.tier = "ssd"
        entry.ssd_path = result["path"]
        entry.size_bytes = int(result["size_mb"] * 1024 * 1024)
        entry.gpu_data = None
        
        del self._gpu_data[key]
        
        self._stats["demotions"] += 1
    
    def _ensure_gpu_space(self, needed_bytes: int) -> None:
        """Ensure enough space in GPU tier by demoting oldest entries."""
        while self._gpu_bytes + needed_bytes > self.max_gpu_bytes and self._gpu_data:
            # Demote oldest (first in OrderedDict)
            oldest_key = next(iter(self._gpu_data))
            self._demote_from_gpu(oldest_key)
    
    def _ensure_ssd_space(self, needed_bytes: int) -> None:
        """Ensure enough space in SSD tier by demoting to R2 or deleting."""
        entries_by_access = sorted(
            [(k, e) for k, e in self._entries.items() if e.tier == "ssd"],
            key=lambda x: x[1].last_accessed
        )
        
        for key, entry in entries_by_access:
            if self._ssd_bytes + needed_bytes <= self.max_ssd_bytes:
                break
            
            # Try to demote to R2
            if self.r2_config:
                self.demote(key, "r2")
            else:
                # No R2, just delete
                self._ssd_cache.delete(key)
                self._ssd_bytes -= entry.size_bytes
                del self._entries[key]
    
    def list_entries(self, tier: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all cache entries, optionally filtered by tier.
        
        Args:
            tier: Optional tier filter ("gpu", "ssd", "r2")
            
        Returns:
            List of entry dicts with metadata
        """
        entries = []
        for key, entry in self._entries.items():
            if tier and entry.tier != tier:
                continue
            
            entries.append({
                "key": key,
                "tier": entry.tier,
                "size_mb": entry.size_bytes / (1024 * 1024),
                "created": datetime.fromtimestamp(entry.created).isoformat(),
                "last_accessed": datetime.fromtimestamp(entry.last_accessed).isoformat(),
                "access_count": entry.access_count,
                "metadata": entry.metadata,
            })
        
        return sorted(entries, key=lambda x: x["last_accessed"], reverse=True)
    
    def delete(self, key: str) -> bool:
        """
        Delete an entry from all tiers.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key not in self._entries:
                return False
            
            entry = self._entries[key]
            
            if entry.tier == "gpu":
                self._gpu_bytes -= entry.size_bytes
                del self._gpu_data[key]
            elif entry.tier == "ssd":
                self._ssd_bytes -= entry.size_bytes
                self._ssd_cache.delete(key)
            # R2 deletion would need additional implementation
            
            del self._entries[key]
            return True
    
    def clear(self, tier: Optional[str] = None) -> int:
        """
        Clear entries from specified tier or all tiers.
        
        Args:
            tier: Optional tier to clear ("gpu", "ssd", "r2", or None for all)
            
        Returns:
            Number of entries cleared
        """
        with self._lock:
            keys_to_delete = []
            
            for key, entry in self._entries.items():
                if tier is None or entry.tier == tier:
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                self.delete(key)
            
            return len(keys_to_delete)
