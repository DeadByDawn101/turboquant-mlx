from setuptools import setup, find_packages

setup(
    name="turboquant-mlx",
    version="2.0.0",
    description="TurboQuant KV Cache Compression for MLX (Apple Silicon)",
    author="RavenX AI",
    author_email="deadbydawn@ravenx.ai",
    url="https://github.com/DeadByDawn101/turboquant-mlx",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "mlx>=0.10.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="mlx apple-silicon kv-cache quantization llm attention",
)
