# Quick Start Guide

## Installation

1. **Install dependencies**:
```bash
cd D:\Code\llm_embedding
pip install -r requirements.txt
pip install -e .
```

2. **Verify installation**:
```bash
python -c "import torch; import transformers; print('OK')"
```

## Running the Albert Experiment

**Run the full pipeline**:
```bash
python experiments/run_albert.py
```

This will:
- Download external datasets (name-dataset, country-state-city)
- Load Albert-base-v2 embeddings
- Extract hierarchical concepts at k=[100, 75, 50, 25, 12, 6]
- Evaluate clusters against named entity databases
- Generate reports and save results

**Expected output locations**:
- `results/communities/albert_communities.json` - Detected communities
- `results/metrics/albert_evaluation_k75.csv` - Evaluation results
- `results/metrics/albert_report_k75.txt` - Summary report

## Expected Results (from Paper)

### Table 1: Number of Communities
| K    | Expected | Your Result |
|------|----------|-------------|
| 100  | 8        | ?           |
| 75   | 133      | ?           |
| 50   | 1058     | ?           |
| 25   | 4442     | ?           |
| 12   | 7718     | ?           |
| 6    | 8626     | ?           |

### Named Entity Evaluation (k=75)
Expected precision scores:
- US/UK/AUS/NZ names: ~0.88 (support ~1011)
- Male names: ~0.85 (support ~946)
- Female names: ~0.87 (support ~552)

## Using the Code Programmatically

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

# Get communities at specific k
k75_communities = extractor.get_communities_at_k(75)

# Evaluate
evaluator = NamedEntityEvaluator()
results_df = evaluator.evaluate_all_clusters(k75_communities)
print(results_df)
```

## Troubleshooting

### Memory Issues
If you run out of memory:
1. Use CPU instead of GPU (slower but less memory)
2. Process fewer k values at a time
3. Reduce vocabulary size for testing

### Download Issues
If datasets don't download automatically:
```python
from src.data.datasets import setup_datasets
setup_datasets(force=True)  # Force re-download
```

### Slow Performance
- The fuzzy graph construction is the slowest part
- For testing, use k_values=[75, 50] instead of all 6 values
- Each k value processes all communities from previous level

## Next Steps

1. **Run T5 experiment**: Modify run_albert.py to use "t5-base"
2. **Test alignment**: Compare Albert vs T5 conceptual structures
3. **Visualize results**: Create UMAP plots of detected communities
4. **Extend evaluation**: Add number ordering, social structure evaluation

## Performance Notes

- Full Albert experiment: ~30-60 minutes on CPU
- Memory usage: ~8-16 GB RAM
- Largest bottleneck: k-NN graph construction at small k values
