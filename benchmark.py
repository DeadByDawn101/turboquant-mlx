#!/usr/bin/env python3
"""
TurboQuant-MLX Benchmark

Compares TurboQuant KV cache compression against standard MLX attention:
- Memory usage
- Attention score accuracy
- Throughput

Usage:
    python benchmark.py [--batch-size 1] [--seq-len 4096] [--head-dim 128]
"""

import mlx.core as mx
import time
import argparse
import math
from typing import Tuple

from turboquant_mlx import (
    TurboQuantKVCache,
    PolarQuantizer,
    QJLSketch,
)


def create_test_data(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Create synthetic test data for benchmarking."""
    # Random query, keys, values
    query = mx.random.normal(shape=(batch_size, num_heads, 1, head_dim))
    keys = mx.random.normal(shape=(batch_size, num_heads, seq_len, head_dim))
    values = mx.random.normal(shape=(batch_size, num_heads, seq_len, head_dim))
    
    return query, keys, values


def standard_attention(
    query: mx.array,
    keys: mx.array,
    values: mx.array,
    mask: mx.array = None,
) -> mx.array:
    """Standard MLX attention implementation."""
    head_dim = query.shape[-1]
    
    # Compute attention scores
    scores = mx.matmul(query, mx.swapaxes(keys, -2, -1))
    scores = scores / math.sqrt(head_dim)
    
    if mask is not None:
        scores = scores + mask
    
    # Softmax
    weights = mx.softmax(scores, axis=-1)
    
    # Output
    output = mx.matmul(weights, values)
    
    return output, scores


def benchmark_memory(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    r_bits: int = 4,
    theta_bits: int = 4,
) -> dict:
    """Benchmark memory usage."""
    print(f"\n{'='*60}")
    print(f"Memory Benchmark")
    print(f"Config: batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")
    print(f"{'='*60}")
    
    _, keys, values = create_test_data(batch_size, num_heads, seq_len, head_dim)
    mx.eval(keys, values)
    
    # Standard: float16 keys + values
    standard_kv_bytes = keys.size * 2 + values.size * 2  # float16 = 2 bytes
    
    # TurboQuant compression
    cache = TurboQuantKVCache(
        head_dim=head_dim,
        num_heads=num_heads,
        r_bits=r_bits,
        theta_bits=theta_bits,
    )
    
    compressed = cache.compress(keys, values)
    usage = cache.memory_usage(compressed)
    
    print(f"\nStandard KV Cache:")
    print(f"  Keys:   {keys.size * 2 / 1024 / 1024:.2f} MB")
    print(f"  Values: {values.size * 2 / 1024 / 1024:.2f} MB")
    print(f"  Total:  {standard_kv_bytes / 1024 / 1024:.2f} MB")
    
    print(f"\nTurboQuant Compressed:")
    print(f"  PolarQuant: {usage['polar_bytes'] / 1024 / 1024:.4f} MB")
    print(f"  QJL:        {usage['qjl_bytes'] / 1024 / 1024:.4f} MB")
    print(f"  Values:     {usage['values_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  Residual:   {usage['residual_bytes'] / 1024 / 1024:.4f} MB")
    print(f"  Total:      {usage['total_compressed'] / 1024 / 1024:.2f} MB")
    
    print(f"\n  Compression Ratio (full KV): {usage['compression_ratio']:.2f}x")
    
    # Key-only compression ratio (what TurboQuant primarily targets)
    key_bytes_standard = keys.size * 2  # float16
    key_bytes_compressed = usage['polar_bytes'] + usage['qjl_bytes'] + usage['residual_bytes']
    key_compression = key_bytes_standard / max(key_bytes_compressed, 1)
    print(f"  Key Compression Ratio: {key_compression:.2f}x")
    
    return usage


def benchmark_accuracy(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    r_bits: int = 4,
    theta_bits: int = 4,
) -> dict:
    """Benchmark attention score accuracy."""
    print(f"\n{'='*60}")
    print(f"Accuracy Benchmark")
    print(f"Config: batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")
    print(f"Quantization: r_bits={r_bits}, theta_bits={theta_bits}")
    print(f"{'='*60}")
    
    query, keys, values = create_test_data(batch_size, num_heads, seq_len, head_dim)
    mx.eval(query, keys, values)
    
    # Standard attention
    std_output, std_scores = standard_attention(query, keys, values)
    mx.eval(std_output, std_scores)
    
    # TurboQuant attention
    cache = TurboQuantKVCache(
        head_dim=head_dim,
        num_heads=num_heads,
        r_bits=r_bits,
        theta_bits=theta_bits,
    )
    
    compressed = cache.compress(keys, values)
    turbo_output, _ = cache.compute_attention(query, compressed)
    mx.eval(turbo_output)
    
    # Compute errors
    output_error = mx.abs(turbo_output - std_output)
    
    max_error = float(mx.max(output_error).item())
    mean_error = float(mx.mean(output_error).item())
    std_val = float(mx.std(std_output).item())
    
    # Relative error
    relative_error = mean_error / (std_val + 1e-10)
    
    print(f"\nOutput Comparison:")
    print(f"  Max Absolute Error:    {max_error:.6f}")
    print(f"  Mean Absolute Error:   {mean_error:.6f}")
    print(f"  Std of Reference:      {std_val:.6f}")
    print(f"  Relative Error:        {relative_error*100:.4f}%")
    
    # Also test PolarQuant reconstruction
    polar = PolarQuantizer(r_bits=r_bits, theta_bits=theta_bits)
    polar_compressed = polar.quantize(keys)
    keys_reconstructed = polar.dequantize(polar_compressed)
    mx.eval(keys_reconstructed)
    
    key_error = keys_reconstructed - keys
    key_mse = float(mx.mean(key_error ** 2).item())
    signal_power = float(mx.mean(keys ** 2).item())
    key_snr = 10 * math.log10(signal_power / (key_mse + 1e-10))
    
    print(f"\nPolarQuant Key Reconstruction:")
    print(f"  MSE:  {key_mse:.6f}")
    print(f"  SNR:  {key_snr:.2f} dB")
    
    return {
        "max_error": max_error,
        "mean_error": mean_error,
        "relative_error": relative_error,
        "key_mse": key_mse,
        "key_snr": key_snr,
    }


def benchmark_speed(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    num_iterations: int = 100,
) -> dict:
    """Benchmark attention throughput."""
    print(f"\n{'='*60}")
    print(f"Speed Benchmark ({num_iterations} iterations)")
    print(f"Config: batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")
    print(f"{'='*60}")
    
    query, keys, values = create_test_data(batch_size, num_heads, seq_len, head_dim)
    mx.eval(query, keys, values)
    
    # Warmup
    for _ in range(5):
        std_output, _ = standard_attention(query, keys, values)
        mx.eval(std_output)
    
    # Standard attention timing
    start = time.perf_counter()
    for _ in range(num_iterations):
        std_output, _ = standard_attention(query, keys, values)
        mx.eval(std_output)
    std_time = (time.perf_counter() - start) / num_iterations * 1000  # ms
    
    # TurboQuant compression (one-time)
    cache = TurboQuantKVCache(
        head_dim=head_dim,
        num_heads=num_heads,
    )
    
    compress_start = time.perf_counter()
    compressed = cache.compress(keys, values)
    mx.eval(compressed.values)  # Force evaluation
    compress_time = (time.perf_counter() - compress_start) * 1000  # ms
    
    # Warmup TurboQuant
    for _ in range(5):
        turbo_output, _ = cache.compute_attention(query, compressed)
        mx.eval(turbo_output)
    
    # TurboQuant attention timing
    start = time.perf_counter()
    for _ in range(num_iterations):
        turbo_output, _ = cache.compute_attention(query, compressed)
        mx.eval(turbo_output)
    turbo_time = (time.perf_counter() - start) / num_iterations * 1000  # ms
    
    print(f"\nTiming Results:")
    print(f"  Standard Attention:    {std_time:.3f} ms/iter")
    print(f"  TurboQuant Attention:  {turbo_time:.3f} ms/iter")
    print(f"  Compression (once):    {compress_time:.3f} ms")
    print(f"\n  Speedup: {std_time/turbo_time:.2f}x")
    
    return {
        "standard_ms": std_time,
        "turbo_ms": turbo_time,
        "compress_ms": compress_time,
        "speedup": std_time / turbo_time,
    }


def run_full_benchmark(args):
    """Run all benchmarks."""
    print("\n" + "="*60)
    print("TurboQuant-MLX Full Benchmark")
    print("First MLX Implementation of TurboQuant")
    print("="*60)
    
    results = {}
    
    # Memory benchmark
    results["memory"] = benchmark_memory(
        args.batch_size,
        args.num_heads,
        args.seq_len,
        args.head_dim,
        args.r_bits,
        args.theta_bits,
    )
    
    # Accuracy benchmark
    results["accuracy"] = benchmark_accuracy(
        args.batch_size,
        args.num_heads,
        args.seq_len,
        args.head_dim,
        args.r_bits,
        args.theta_bits,
    )
    
    # Speed benchmark
    results["speed"] = benchmark_speed(
        args.batch_size,
        args.num_heads,
        args.seq_len,
        args.head_dim,
        args.num_iterations,
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nCompression Ratio:    {results['memory']['compression_ratio']:.2f}x")
    print(f"Relative Error:       {results['accuracy']['relative_error']*100:.4f}%")
    print(f"Key Reconstruction:   {results['accuracy']['key_snr']:.2f} dB SNR")
    print(f"Attention Speedup:    {results['speed']['speedup']:.2f}x")
    
    # Long context simulation
    if args.long_context:
        print("\n" + "="*60)
        print("Long Context Simulation")
        print("="*60)
        
        for seq_len in [8192, 16384, 32768, 65536]:
            try:
                usage = benchmark_memory(
                    args.batch_size,
                    args.num_heads,
                    seq_len,
                    args.head_dim,
                    args.r_bits,
                    args.theta_bits,
                )
            except Exception as e:
                print(f"  seq_len={seq_len}: FAILED ({e})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="TurboQuant-MLX Benchmark")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--head-dim", type=int, default=128, help="Dimension per head")
    parser.add_argument("--r-bits", type=int, default=4, help="Radius quantization bits")
    parser.add_argument("--theta-bits", type=int, default=4, help="Angle quantization bits")
    parser.add_argument("--num-iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--long-context", action="store_true", help="Run long context tests")
    
    args = parser.parse_args()
    
    # Run benchmarks
    run_full_benchmark(args)


if __name__ == "__main__":
    main()
