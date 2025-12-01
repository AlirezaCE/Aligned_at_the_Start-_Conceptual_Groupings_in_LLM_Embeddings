"""
Run concept extraction experiment on Albert model.

This script:
1. Loads Albert embeddings
2. Extracts hierarchical concepts
3. Evaluates against named entity datasets
4. Saves results and generates reports
"""

import os
import sys
import logging
from pathlib import Path
import yaml
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.embeddings.loader import EmbeddingLoader
from src.extraction.concept_extractor import ConceptExtractor
from src.evaluation.named_entities import NamedEntityEvaluator
from src.data.datasets import setup_datasets

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run Albert experiment."""
    logger.info("=" * 80)
    logger.info("Albert Concept Extraction Experiment")
    logger.info("=" * 80)

    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup paths
    results_dir = Path(config['output']['communities_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = Path(config['output']['metrics_dir'])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Setup datasets
    logger.info("\nStep 1: Setting up external datasets...")
    setup_datasets(config['data']['raw_dir'])

    # Load Albert embeddings
    logger.info("\nStep 2: Loading Albert embeddings...")
    model_name = config['models']['albert']['name']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    loader = EmbeddingLoader(model_name, device=device)

    embeddings = loader.get_embeddings()
    vocabulary = loader.get_vocabulary()

    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Vocabulary size: {len(vocabulary)}")

    # Extract concepts
    logger.info("\nStep 3: Extracting concepts...")
    extractor = ConceptExtractor(
        embeddings=embeddings,
        vocabulary=vocabulary,
        k_values=config['k_values'],
        random_state=config['computation']['random_seed']
    )

    hierarchical_communities = extractor.extract_concepts()

    # Print summary
    logger.info("\nExtraction Summary:")
    summary = extractor.get_summary()
    for k, stats in summary['results_by_k'].items():
        logger.info(f"k={k}: {stats['n_communities']} communities, "
                   f"mean size={stats['mean_size']:.1f}")

    # Save extraction results
    extraction_results_path = results_dir / "albert_communities.json"
    extractor.save_results(str(extraction_results_path))

    # Evaluate with named entities
    logger.info("\nStep 4: Evaluating with named entities...")
    evaluator = NamedEntityEvaluator(
        data_dir=config['data']['raw_dir'],
        min_cluster_size=config['evaluation']['min_cluster_size']
    )

    # Evaluate at k=75 (as in paper)
    k_to_evaluate = 75
    communities_k75 = extractor.get_communities_at_k(k_to_evaluate)
    logger.info(f"\nEvaluating {len(communities_k75)} communities at k={k_to_evaluate}")

    eval_df = evaluator.evaluate_all_clusters(communities_k75)

    # Save evaluation results
    eval_results_path = metrics_dir / f"albert_evaluation_k{k_to_evaluate}.csv"
    eval_df.to_csv(eval_results_path, index=False)
    logger.info(f"Saved evaluation results to: {eval_results_path}")

    # Generate report
    report = evaluator.generate_report(eval_df)
    report_path = metrics_dir / f"albert_report_k{k_to_evaluate}.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    print("\n" + report)

    # Compare with paper results
    logger.info("\n" + "=" * 80)
    logger.info("Comparison with Paper Results (Table 1)")
    logger.info("=" * 80)
    logger.info(f"Expected number of communities at different k values:")
    logger.info(f"  k=100: Expected ~8, Got {summary['results_by_k'].get(100, {}).get('n_communities', 'N/A')}")
    logger.info(f"  k=75:  Expected ~133, Got {summary['results_by_k'].get(75, {}).get('n_communities', 'N/A')}")
    logger.info(f"  k=50:  Expected ~1058, Got {summary['results_by_k'].get(50, {}).get('n_communities', 'N/A')}")
    logger.info(f"  k=25:  Expected ~4442, Got {summary['results_by_k'].get(25, {}).get('n_communities', 'N/A')}")

    logger.info("\n" + "=" * 80)
    logger.info("Experiment completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
