"""
TurboQuant Ollama Integration

Provides a TurboQuantOllamaClient that wraps the OpenAI-compatible API
for Ollama at http://localhost:11434/v1.

**Important Note**: Ollama manages its own KV cache internally - this wrapper
provides stats/monitoring and a consistent interface. True KV compression
with TurboQuant requires mlx-lm or llama.cpp backends where we can intercept
the actual attention mechanism.

This integration tracks:
- Token count (input/output)
- Estimated memory savings (based on TurboQuant compression ratios)
- Timing statistics

Author: RavenX AI
License: MIT
"""

import os
import time
from typing import Any, Dict, List, Optional, Union, Generator
from dataclasses import dataclass, field

# Try importing openai - it's optional
try:
    import openai
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None


@dataclass
class OllamaStats:
    """Statistics tracked by TurboQuantOllamaClient."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    total_latency_ms: float = 0.0
    
    # Estimated memory savings (based on TurboQuant compression)
    estimated_kv_bytes_uncompressed: int = 0
    estimated_kv_bytes_compressed: int = 0
    
    # Config
    compression_ratio: float = 4.0  # Default TurboQuant ratio
    bytes_per_token_kv: int = 512   # Approximate KV bytes per token (depends on model)
    
    def reset(self):
        """Reset all statistics."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.total_latency_ms = 0.0
        self.estimated_kv_bytes_uncompressed = 0
        self.estimated_kv_bytes_compressed = 0
    
    def update(self, input_tokens: int, output_tokens: int, latency_ms: float):
        """Update stats with a new request."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1
        self.total_latency_ms += latency_ms
        
        # Estimate KV memory (context = input + output accumulated)
        total_tokens = input_tokens + output_tokens
        uncompressed = total_tokens * self.bytes_per_token_kv
        compressed = int(uncompressed / self.compression_ratio)
        
        self.estimated_kv_bytes_uncompressed += uncompressed
        self.estimated_kv_bytes_compressed += compressed
    
    def summary(self) -> Dict[str, Any]:
        """Return summary statistics."""
        savings_mb = (
            self.estimated_kv_bytes_uncompressed - self.estimated_kv_bytes_compressed
        ) / (1024 * 1024)
        
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_requests": self.total_requests,
            "avg_latency_ms": (
                self.total_latency_ms / self.total_requests
                if self.total_requests > 0 else 0
            ),
            "estimated_memory_savings_mb": savings_mb,
            "compression_ratio": self.compression_ratio,
        }


class TurboQuantOllamaClient:
    """
    Ollama client with TurboQuant stats tracking.
    
    Wraps the OpenAI-compatible API to provide:
    - `.chat()` method for chat completions
    - `.generate()` method for text completions
    - `.stats()` method for memory savings estimates
    
    Note: Ollama manages KV cache internally. This wrapper provides
    monitoring and a consistent interface. True KV compression requires
    mlx-lm or llama.cpp backends where we can intercept attention.
    
    Args:
        base_url: Ollama API URL (default: http://localhost:11434/v1)
        api_key: API key (default: "ollama" - not required for local)
        compression_ratio: Expected TurboQuant compression ratio (default: 4.0)
        bytes_per_token_kv: Estimated KV bytes per token (default: 512)
    
    Example:
        >>> client = TurboQuantOllamaClient()
        >>> response = client.chat("qwen2.5:7b", messages=[
        ...     {"role": "user", "content": "Hello!"}
        ... ])
        >>> print(response.choices[0].message.content)
        >>> print(client.stats())
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        compression_ratio: float = 4.0,
        bytes_per_token_kv: int = 512,
        **kwargs,
    ):
        if not HAS_OPENAI:
            raise ImportError(
                "openai package is required for TurboQuantOllamaClient. "
                "Install it with: pip install openai"
            )
        
        self.base_url = base_url
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            **kwargs,
        )
        
        self._stats = OllamaStats(
            compression_ratio=compression_ratio,
            bytes_per_token_kv=bytes_per_token_kv,
        )
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """
        Send a chat completion request to Ollama.
        
        Args:
            model: Model name (e.g., "qwen2.5:7b", "llama3.2:3b")
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response
            **kwargs: Additional parameters passed to the API
            
        Returns:
            OpenAI ChatCompletion response
        """
        start_time = time.time()
        
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        if not stream:
            # Extract token counts from response
            input_tokens = getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0
            output_tokens = getattr(response.usage, 'completion_tokens', 0) if response.usage else 0
            
            # Estimate if usage not provided
            if input_tokens == 0:
                input_tokens = sum(len(m.get('content', '')) // 4 for m in messages)
            if output_tokens == 0 and response.choices:
                content = response.choices[0].message.content or ""
                output_tokens = len(content) // 4
            
            self._stats.update(input_tokens, output_tokens, latency_ms)
        
        return response
    
    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """
        Send a text completion request to Ollama.
        
        Uses the chat API under the hood with a single user message.
        
        Args:
            model: Model name
            prompt: Text prompt
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            OpenAI ChatCompletion response
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(model, messages, stream=stream, **kwargs)
    
    def stats(self) -> Dict[str, Any]:
        """
        Get current statistics including memory savings estimates.
        
        Returns:
            Dict with token counts, latency, and memory savings estimate
        """
        return self._stats.summary()
    
    def reset_stats(self):
        """Reset all statistics to zero."""
        self._stats.reset()
    
    @property
    def client(self) -> "OpenAI":
        """Access the underlying OpenAI client for advanced usage."""
        return self._client


def patch_ollama_env(
    num_parallel: int = 4,
    num_ctx: int = 8192,
    flash_attention: bool = True,
    keep_alive: str = "24h",
) -> Dict[str, str]:
    """
    Set OLLAMA_* environment variables for optimal performance.
    
    These settings help Ollama perform better with long contexts
    and multiple concurrent requests.
    
    Args:
        num_parallel: Number of parallel requests (default: 4)
        num_ctx: Context window size (default: 8192)
        flash_attention: Enable flash attention (default: True)
        keep_alive: How long to keep model loaded (default: "24h")
        
    Returns:
        Dict of environment variables that were set
        
    Example:
        >>> from turboquant_mlx.ollama_patch import patch_ollama_env
        >>> env = patch_ollama_env(num_ctx=32768)
        >>> # Now start Ollama with optimized settings
    """
    env_vars = {
        "OLLAMA_NUM_PARALLEL": str(num_parallel),
        "OLLAMA_FLASH_ATTENTION": "1" if flash_attention else "0",
        "OLLAMA_KEEP_ALIVE": keep_alive,
    }
    
    # num_ctx is set per-model, but we can set a default
    if num_ctx:
        env_vars["OLLAMA_NUM_CTX"] = str(num_ctx)
    
    # Apply to environment
    for key, value in env_vars.items():
        os.environ[key] = value
    
    return env_vars


__all__ = [
    "TurboQuantOllamaClient",
    "OllamaStats", 
    "patch_ollama_env",
    "HAS_OPENAI",
]
