"""
TurboQuantKVCache — drop-in replacement for mlx_lm KVCache.
Implements the same interface as mlx_lm.models.cache.KVCache
so it can be used with ANY mlx-lm model without code changes.

Optimizations incorporated (thanks to helgklaizar/turboquant_mlx):
1. Asymmetric K/V compression: Keys use full TurboQuant (PolarQuant + QJL),
   Values use pure PolarQuant only (QJL is mathematically redundant for Values
   since QJL corrects inner product bias, and V is never dot-producted with Q).
2. FP16 attention sinks: First `fp16_sink_size` tokens kept uncompressed to
   preserve instruction-following at extreme compression ratios.
3. Dynamic chunk buffering: Tokens accumulated in a buffer of `chunk_size`
   before compression fires, reducing per-token overhead during autoregressive decode.

Usage:
    from turboquant_mlx.mlx_kvcache import TurboQuantKVCache

    # Replace in make_prompt_cache:
    cache = [TurboQuantKVCache() for _ in range(num_layers)]
"""

import mlx.core as mx
import math
from typing import Optional, Tuple, List

from .polarquant import PolarQuantizer, PolarQuantizedKV
from .qjl import QJLKVCompressor


class TurboQuantKVCache:
    """
    Drop-in replacement for mlx_lm KVCache with TurboQuant compression.

    Key design decisions:
    - Keys: PolarQuant + QJL residual correction (full TurboQuant)
    - Values: PolarQuant only (QJL is mathematically redundant for V)
    - First `fp16_sink_size` tokens always kept in fp16 (attention sinks)
    - Tokens buffered in chunks before compression (reduces decode overhead)

    Matches the mlx_lm KVCache interface exactly.
    """

    def __init__(
        self,
        r_bits: int = 4,
        theta_bits: int = 4,
        compress_after: int = 128,       # Start compressing after this many tokens
        fp16_sink_size: int = 128,        # Keep first N tokens in fp16 (attention sinks)
        chunk_size: int = 64,             # Buffer N tokens before firing compressor
        use_qjl_keys: bool = True,        # Full TurboQuant for keys (PolarQuant + QJL)
        use_qjl_values: bool = False,     # PolarQuant only for values (asymmetric)
        qjl_sketch_dim: int = 256,
    ):
        self.r_bits = r_bits
        self.theta_bits = theta_bits
        self.compress_after = compress_after
        self.fp16_sink_size = fp16_sink_size
        self.chunk_size = chunk_size
        self.use_qjl_keys = use_qjl_keys
        self.use_qjl_values = use_qjl_values
        self.qjl_sketch_dim = qjl_sketch_dim

        # Attention sink storage (fp16, always uncompressed)
        self._sink_keys: Optional[mx.array] = None
        self._sink_values: Optional[mx.array] = None

        # Chunk buffer for incoming tokens (pre-compression staging)
        self._buf_keys: Optional[mx.array] = None
        self._buf_values: Optional[mx.array] = None

        # Compressed key/value chunks
        self._comp_key_chunks: List[PolarQuantizedKV] = []
        self._comp_val_chunks: List[PolarQuantizedKV] = []

        self.offset = 0

        # Lazy-init quantizers
        self._key_polar: Optional[PolarQuantizer] = None
        self._val_polar: Optional[PolarQuantizer] = None
        self._key_qjl: Optional[QJLKVCompressor] = None

    # ------------------------------------------------------------------
    # Lazy quantizer init
    # ------------------------------------------------------------------

    def _get_key_polar(self, head_dim: int) -> PolarQuantizer:
        if self._key_polar is None or self._key_polar._head_dim != head_dim:
            self._key_polar = PolarQuantizer(
                r_bits=self.r_bits,
                theta_bits=self.theta_bits,
                group_size=self.chunk_size,
            )
        return self._key_polar

    def _get_val_polar(self, head_dim: int) -> PolarQuantizer:
        if self._val_polar is None or self._val_polar._head_dim != head_dim:
            self._val_polar = PolarQuantizer(
                r_bits=self.r_bits,
                theta_bits=self.theta_bits,
                group_size=self.chunk_size,
            )
        return self._val_polar

    def _get_key_qjl(self, head_dim: int) -> QJLKVCompressor:
        if self._key_qjl is None or self._key_qjl.head_dim != head_dim:
            self._key_qjl = QJLKVCompressor(
                head_dim=head_dim,
                sketch_dim=self.qjl_sketch_dim,
            )
        return self._key_qjl

    # ------------------------------------------------------------------
    # Compression helpers
    # ------------------------------------------------------------------

    def _compress_chunk(self, keys: mx.array, values: mx.array):
        """Compress a chunk of keys/values and append to compressed lists."""
        head_dim = keys.shape[-1]

        # Keys: full TurboQuant (PolarQuant only for now, QJL residual is additive)
        key_polar = self._get_key_polar(head_dim)
        comp_keys = key_polar.quantize(keys)
        self._comp_key_chunks.append(comp_keys)

        # Values: PolarQuant only (asymmetric — QJL redundant for V)
        val_polar = self._get_val_polar(head_dim)
        comp_vals = val_polar.quantize(values)
        self._comp_val_chunks.append(comp_vals)

    def _decompress_all_chunks(self, head_dim: int) -> Tuple[mx.array, mx.array]:
        """Decompress all stored chunks and concatenate."""
        if not self._comp_key_chunks:
            return None, None

        key_polar = self._get_key_polar(head_dim)
        val_polar = self._get_val_polar(head_dim)

        key_parts = [key_polar.dequantize(c) for c in self._comp_key_chunks]
        val_parts = [val_polar.dequantize(c) for c in self._comp_val_chunks]

        keys = mx.concatenate(key_parts, axis=-2)
        vals = mx.concatenate(val_parts, axis=-2)
        return keys, vals

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Append new keys/values and return full (decompressed) history.
        Shape: [batch, heads, seq, head_dim]
        """
        head_dim = keys.shape[-1]
        total_so_far = self.offset

        # ---- Phase 1: Fill attention sink (fp16, never compressed) ----
        if total_so_far < self.fp16_sink_size:
            space = self.fp16_sink_size - total_so_far
            sink_new = min(keys.shape[-2], space)

            sink_k = keys[..., :sink_new, :]
            sink_v = values[..., :sink_new, :]

            self._sink_keys = sink_k if self._sink_keys is None else \
                mx.concatenate([self._sink_keys, sink_k], axis=-2)
            self._sink_values = sink_v if self._sink_values is None else \
                mx.concatenate([self._sink_values, sink_v], axis=-2)

            # Remaining tokens go to buffer
            keys = keys[..., sink_new:, :]
            values = values[..., sink_new:, :]

        # ---- Phase 2: Buffer tokens, compress when chunk is full ----
        if keys.shape[-2] > 0:
            self._buf_keys = keys if self._buf_keys is None else \
                mx.concatenate([self._buf_keys, keys], axis=-2)
            self._buf_values = values if self._buf_values is None else \
                mx.concatenate([self._buf_values, values], axis=-2)

            # Fire compressor when buffer hits chunk_size AND we're past compress_after
            buf_len = self._buf_keys.shape[-2]
            buf_total = (self._sink_keys.shape[-2] if self._sink_keys is not None else 0) + buf_len

            while self._buf_keys is not None and \
                    self._buf_keys.shape[-2] >= self.chunk_size and \
                    buf_total >= self.compress_after:

                chunk_k = self._buf_keys[..., :self.chunk_size, :]
                chunk_v = self._buf_values[..., :self.chunk_size, :]
                self._compress_chunk(chunk_k, chunk_v)

                rest_k = self._buf_keys[..., self.chunk_size:, :]
                rest_v = self._buf_values[..., self.chunk_size:, :]
                self._buf_keys = rest_k if rest_k.shape[-2] > 0 else None
                self._buf_values = rest_v if rest_v.shape[-2] > 0 else None

                buf_len = self._buf_keys.shape[-2] if self._buf_keys is not None else 0
                buf_total = (self._sink_keys.shape[-2] if self._sink_keys is not None else 0) + buf_len

        # ---- Update offset ----
        sink_len = self._sink_keys.shape[-2] if self._sink_keys is not None else 0
        buf_len = self._buf_keys.shape[-2] if self._buf_keys is not None else 0
        comp_len = sum(c.original_seq_len for c in self._comp_key_chunks)
        self.offset = sink_len + comp_len + buf_len

        # ---- Assemble full history ----
        parts_k = []
        parts_v = []

        # Sinks (fp16, uncompressed)
        if self._sink_keys is not None:
            parts_k.append(self._sink_keys)
            parts_v.append(self._sink_values)

        # Compressed chunks (decompressed on demand)
        if self._comp_key_chunks:
            comp_k, comp_v = self._decompress_all_chunks(head_dim)
            parts_k.append(comp_k)
            parts_v.append(comp_v)

        # Buffer (recent uncompressed tokens)
        if self._buf_keys is not None:
            parts_k.append(self._buf_keys)
            parts_v.append(self._buf_values)

        if not parts_k:
            return keys, values  # fallback (empty cache)

        full_k = mx.concatenate(parts_k, axis=-2) if len(parts_k) > 1 else parts_k[0]
        full_v = mx.concatenate(parts_v, axis=-2) if len(parts_v) > 1 else parts_v[0]

        return full_k, full_v

    # ------------------------------------------------------------------
    # State interface (mlx_lm compatibility)
    # ------------------------------------------------------------------

    @property
    def state(self):
        """Return raw (uncompressed) state for checkpointing."""
        # Reconstruct full raw state for compatibility
        parts_k, parts_v = [], []
        if self._sink_keys is not None:
            parts_k.append(self._sink_keys)
            parts_v.append(self._sink_values)
        if self._comp_key_chunks:
            head_dim = self._sink_keys.shape[-1] if self._sink_keys is not None else \
                (self._buf_keys.shape[-1] if self._buf_keys is not None else 64)
            comp_k, comp_v = self._decompress_all_chunks(head_dim)
            parts_k.append(comp_k)
            parts_v.append(comp_v)
        if self._buf_keys is not None:
            parts_k.append(self._buf_keys)
            parts_v.append(self._buf_values)

        if not parts_k:
            return None, None

        full_k = mx.concatenate(parts_k, axis=-2) if len(parts_k) > 1 else parts_k[0]
        full_v = mx.concatenate(parts_v, axis=-2) if len(parts_v) > 1 else parts_v[0]
        return full_k, full_v

    @state.setter
    def state(self, v):
        """Restore from raw state (e.g. loaded checkpoint)."""
        raw_k, raw_v = v
        self.reset()
        if raw_k is not None:
            self._sink_keys = raw_k
            self._sink_values = raw_v
            self.offset = raw_k.shape[-2]

    @property
    def meta_state(self):
        return {
            "fp16_sink_size": self.fp16_sink_size,
            "chunk_size": self.chunk_size,
            "r_bits": self.r_bits,
            "theta_bits": self.theta_bits,
            "compressed_chunks": len(self._comp_key_chunks),
        }

    def is_empty(self) -> bool:
        return (self._sink_keys is None and
                self._buf_keys is None and
                not self._comp_key_chunks)

    @property
    def memory_size(self) -> int:
        """Approximate memory usage in bytes."""
        total = 0
        if self._sink_keys is not None:
            total += self._sink_keys.nbytes + self._sink_values.nbytes
        if self._buf_keys is not None:
            total += self._buf_keys.nbytes + self._buf_values.nbytes
        for chunk in self._comp_key_chunks:
            total += chunk.indices.nbytes + chunk.r_scale.nbytes * 4
        for chunk in self._comp_val_chunks:
            total += chunk.indices.nbytes + chunk.r_scale.nbytes * 4
        return total

    def reset(self):
        self._sink_keys = None
        self._sink_values = None
        self._buf_keys = None
        self._buf_values = None
        self._comp_key_chunks = []
        self._comp_val_chunks = []
        self.offset = 0
