"""
Concept extraction algorithm (Algorithm 1 from the paper).

Implements hierarchical clustering detection at multiple k-NN granularities.
"""

import numpy as np
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import json

from ..graph.fuzzy_graph import FuzzyGraphBuilder
from ..clustering.louvain import LouvainDetector

logger = logging.getLogger(__name__)


class ConceptExtractor:
    """
    Extract hierarchical conceptual groupings from embeddings.

    Implements Algorithm 1: Iteratively builds k-NN graphs at different
    granularities and applies Louvain clustering detection.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        vocabulary: List[str],
        k_values: List[int] = [100, 75, 50, 25, 12, 6],
        random_state: int = 42
    ):
        """
        Initialize the concept extractor.

        Args:
            embeddings: Array of shape (vocab_size, embedding_dim)
            vocabulary: List of token strings
            k_values: List of k values for hierarchical extraction
            random_state: Random seed for reproducibility
        """
        self.embeddings = embeddings
        self.vocabulary = vocabulary
        self.k_values = sorted(k_values, reverse=True)  # Descending order
        self.random_state = random_state
        self.vocab_size = len(vocabulary)

        logger.info(f"Initialized ConceptExtractor with {self.vocab_size} tokens")
        logger.info(f"K values: {self.k_values}")

        self.hierarchical_communities = {}

    def extract_concepts(self) -> Dict:
        """
        Extract hierarchical concepts using Algorithm 1.

        Returns:
            Dictionary with results for each k value
        """
        logger.info("=" * 80)
        logger.info("Starting concept extraction (Algorithm 1)")
        logger.info("=" * 80)

        # Initialize: entire vocabulary is one clustering
        all_indices = list(range(self.vocab_size))
        current_communities = [all_indices]

        for k in self.k_values:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Processing k={k}")
            logger.info(f"{'=' * 80}")

            new_communities = []

            # Process each clustering from previous iteration
            for comm_idx, community_indices in enumerate(tqdm(
                current_communities,
                desc=f"k={k} communities"
            )):
                if len(community_indices) <= k:
                    # Community too small for k-NN, keep as is
                    new_communities.append(community_indices)
                    continue

                # Extract sub-communities from this clustering
                sub_communities = self._process_community(community_indices, k)
                new_communities.extend(sub_communities)

            # Store results for this k
            self.hierarchical_communities[k] = {
                'k': k,
                'n_communities': len(new_communities),
                'communities': new_communities,
                'sizes': [len(c) for c in new_communities]
            }

            logger.info(f"k={k}: Found {len(new_communities)} communities")
            logger.info(f"  Sizes - min: {min(len(c) for c in new_communities)}, "
                       f"max: {max(len(c) for c in new_communities)}, "
                       f"mean: {np.mean([len(c) for c in new_communities]):.1f}")

            # Update for next iteration
            current_communities = new_communities

        logger.info("\n" + "=" * 80)
        logger.info("Concept extraction completed!")
        logger.info("=" * 80)

        return self.hierarchical_communities

    def _process_community(self, indices: List[int], k: int) -> List[List[int]]:
        """
        Process a single clustering: build k-NN graph and detect sub-communities.

        Args:
            indices: List of token indices in this clustering
            k: Number of nearest neighbors

        Returns:
            List of sub-clustering indices
        """
        # Extract embeddings for this clustering
        sub_embeddings = self.embeddings[indices]

        # Build fuzzy k-NN graph
        graph_builder = FuzzyGraphBuilder(sub_embeddings, k)
        adjacency = graph_builder.build_knn_graph()

        # Detect communities
        detector = LouvainDetector(adjacency, random_state=self.random_state)
        partition = detector.detect_communities()

        # Convert partition to list of sub-communities
        # Map local indices back to global indices
        sub_communities = detector.partition_to_list(partition)

        # Convert local indices to global indices
        global_communities = []
        for sub_comm in sub_communities:
            global_indices = [indices[local_idx] for local_idx in sub_comm]
            global_communities.append(global_indices)

        return global_communities

    def get_communities_at_k(self, k: int) -> List[Dict]:
        """
        Get communities detected at a specific k value.

        Args:
            k: K value

        Returns:
            List of clustering dictionaries with indices and tokens
        """
        if k not in self.hierarchical_communities:
            raise ValueError(f"k={k} not found. Available: {list(self.hierarchical_communities.keys())}")

        communities = []
        for idx, comm_indices in enumerate(self.hierarchical_communities[k]['communities']):
            communities.append({
                'id': idx,
                'k': k,
                'size': len(comm_indices),
                'indices': comm_indices,
                'tokens': [self.vocabulary[i] for i in comm_indices]
            })

        return communities

    def get_summary(self) -> Dict:
        """
        Get summary statistics of extracted concepts.

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'vocab_size': self.vocab_size,
            'k_values': self.k_values,
            'results_by_k': {}
        }

        for k in self.k_values:
            if k in self.hierarchical_communities:
                data = self.hierarchical_communities[k]
                summary['results_by_k'][k] = {
                    'n_communities': data['n_communities'],
                    'min_size': min(data['sizes']),
                    'max_size': max(data['sizes']),
                    'mean_size': np.mean(data['sizes']),
                    'median_size': np.median(data['sizes'])
                }

        return summary

    def save_results(self, filepath: str):
        """
        Save extraction results to JSON file.

        Args:
            filepath: Output file path
        """
        # Convert numpy types to Python types for JSON serialization
        output = {
            'vocab_size': int(self.vocab_size),
            'k_values': [int(k) for k in self.k_values],
            'communities': {}
        }

        for k in self.k_values:
            if k in self.hierarchical_communities:
                communities_list = self.get_communities_at_k(k)
                # Don't save full token lists (too large), just indices
                output['communities'][int(k)] = [
                    {
                        'id': c['id'],
                        'size': c['size'],
                        'indices': c['indices'][:100]  # Save first 100 only
                    }
                    for c in communities_list
                ]

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved results to {filepath}")
