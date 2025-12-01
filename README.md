# LLM Embedding Concepts

Implementation of the paper "Aligned at the Start: Conceptual Groupings in LLM Embeddings" by Khatir et al. (2025).

## Overview

This project implements a method to extract and analyze conceptual structures within the input embedding space of Large Language Models (LLMs). The approach uses fuzzy graph construction and Louvain community detection to identify hierarchical conceptual clusters.

## Key Features

- **Fuzzy Graph Construction**: UMAP-based k-NN graph with weighted edges
- **Hierarchical Community Detection**: Louvain algorithm for multi-level concept extraction
- **Named Entity Evaluation**: Precision-based evaluation against external databases
- **LLM-LLM Alignment**: Cross-model conceptual alignment analysis
- **Multiple LLM Support**: Albert, T5, and extensible to other models

## Installation

```bash
# Clone the repository
cd llm_embedding

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Project Structure

```
llm_embedding/
├── config/              # Configuration files
├── src/                 # Source code
│   ├── embeddings/      # Embedding extraction
│   ├── graph/           # Fuzzy graph construction
│   ├── community/       # Community detection
│   ├── extraction/      # Concept extraction algorithm
│   ├── evaluation/      # Evaluation metrics
│   ├── data/            # Dataset handling
│   └── utils/           # Utilities
├── experiments/         # Experiment scripts
├── notebooks/           # Analysis notebooks
├── data/                # Downloaded datasets
└── results/             # Generated outputs
```

## Usage

### Quick Start

```python
from src.embeddings.loader import EmbeddingLoader
from src.extraction.concept_extractor import ConceptExtractor
from src.evaluation.named_entities import NamedEntityEvaluator

# Load embeddings
loader = EmbeddingLoader("albert-base-v2")
embeddings = loader.get_embeddings()
vocabulary = loader.get_vocabulary()

# Extract concepts
extractor = ConceptExtractor(embeddings, vocabulary)
communities = extractor.extract_concepts()

# Evaluate
evaluator = NamedEntityEvaluator()
results = evaluator.evaluate_all_clusters(communities)
print(results)
```

### Running Experiments

```bash
# Run Albert experiment
python experiments/run_albert.py

# Run T5 experiment
python experiments/run_t5.py

# Run alignment analysis
python experiments/run_alignment.py
```

## Paper Reference

```bibtex
@article{khatir2025aligned,
  title={Aligned at the Start: Conceptual Groupings in LLM Embeddings},
  author={Khatir, Mehrdad and Kabra, Sanchit and Reddy, Chandan K.},
  journal={arXiv preprint arXiv:2406.05315},
  year={2025}
}
```

## License

This implementation is for research purposes.
