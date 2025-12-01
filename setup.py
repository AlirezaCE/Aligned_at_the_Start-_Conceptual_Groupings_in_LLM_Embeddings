from setuptools import setup, find_packages

setup(
    name="llm-embedding-concepts",
    version="0.1.0",
    description="Implementation of 'Aligned at the Start: Conceptual Groupings in LLM Embeddings'",
    author="Research Implementation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "python-louvain>=0.16",
        "networkx>=3.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "umap-learn>=0.5.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)
