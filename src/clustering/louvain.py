"""
Louvain clustering detection wrapper.

Implements modularity-based clustering detection using the Louvain algorithm.
"""

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple
import logging
import community as community_louvain

logger = logging.getLogger(__name__)


class LouvainDetector:
    """
    Wrapper for Louvain clustering detection algorithm.

    The Louvain method optimizes modularity:
        Q = (1/2m) Σ [Aij − kikj/2m] δ(ci, cj)

    where:
    - Aij is the edge weight between i and j
    - ki is the sum of weights of edges attached to vertex i
    - m is the sum of all edge weights
    - δ(ci, cj) is 1 if i and j are in the same clustering, 0 otherwise
    """

    def __init__(self, adjacency_matrix: csr_matrix, random_state: int = 42):
        """
        Initialize the Louvain detector.

        Args:
            adjacency_matrix: Sparse adjacency matrix
            random_state: Random seed for reproducibility
        """
        self.adjacency_matrix = adjacency_matrix
        self.random_state = random_state
        self.n_nodes = adjacency_matrix.shape[0]

        # Convert to NetworkX graph
        logger.info(f"Converting adjacency matrix to NetworkX graph ({self.n_nodes} nodes)...")
        self.graph = self._to_networkx()

    def _to_networkx(self) -> nx.Graph:
        """
        Convert sparse adjacency matrix to NetworkX graph.

        Returns:
            NetworkX graph with weighted edges
        """
        # Create graph from sparse matrix
        G = nx.from_scipy_sparse_array(self.adjacency_matrix)

        return G

    def detect_communities(self, resolution: float = 1.0) -> Dict[int, int]:
        """
        Run Louvain clustering detection.

        Args:
            resolution: Resolution parameter for modularity (default 1.0)

        Returns:
            Dictionary mapping node index to clustering ID
        """
        logger.info("Running Louvain clustering detection...")

        # Run Louvain algorithm
        partition = community_louvain.best_partition(
            self.graph,
            resolution=resolution,
            random_state=self.random_state
        )

        n_communities = len(set(partition.values()))
        logger.info(f"Found {n_communities} communities")

        return partition

    def compute_modularity(self, partition: Dict[int, int]) -> float:
        """
        Compute modularity Q for a given partition.

        Args:
            partition: Dictionary mapping node to clustering

        Returns:
            Modularity score
        """
        modularity = community_louvain.modularity(partition, self.graph)
        return modularity

    def get_community_sizes(self, partition: Dict[int, int]) -> Dict[int, int]:
        """
        Get the size of each clustering.

        Args:
            partition: Dictionary mapping node to clustering

        Returns:
            Dictionary mapping clustering ID to size
        """
        sizes = {}
        for node, comm_id in partition.items():
            sizes[comm_id] = sizes.get(comm_id, 0) + 1

        return sizes

    def partition_to_list(self, partition: Dict[int, int]) -> List[List[int]]:
        """
        Convert partition dictionary to list of communities.

        Args:
            partition: Dictionary mapping node to clustering

        Returns:
            List where each element is a list of node indices in that clustering
        """
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)

        return list(communities.values())


def hierarchical_louvain(
    adjacency_matrix: csr_matrix,
    max_levels: int = 3,
    random_state: int = 42
) -> List[Dict[int, int]]:
    """
    Run hierarchical Louvain detection.

    Repeatedly applies Louvain to create a hierarchy of communities.

    Args:
        adjacency_matrix: Sparse adjacency matrix
        max_levels: Maximum number of hierarchical levels
        random_state: Random seed

    Returns:
        List of partitions, one per level
    """
    levels = []
    current_matrix = adjacency_matrix

    for level in range(max_levels):
        logger.info(f"Computing hierarchical level {level + 1}...")

        detector = LouvainDetector(current_matrix, random_state)
        partition = detector.detect_communities()
        modularity = detector.compute_modularity(partition)

        logger.info(f"Level {level + 1}: {len(set(partition.values()))} communities, modularity={modularity:.4f}")

        levels.append(partition)

        # Check if we should stop (only one clustering or no improvement)
        if len(set(partition.values())) == 1:
            logger.info("Single clustering detected, stopping hierarchy")
            break

        # Aggregate for next level
        current_matrix = _aggregate_communities(current_matrix, partition)

    return levels


def _aggregate_communities(
    adjacency_matrix: csr_matrix,
    partition: Dict[int, int]
) -> csr_matrix:
    """
    Aggregate communities into super-nodes for next hierarchical level.

    Args:
        adjacency_matrix: Current adjacency matrix
        partition: Community assignment

    Returns:
        Aggregated adjacency matrix
    """
    n_communities = len(set(partition.values()))

    # Create mapping from old nodes to new super-nodes
    comm_list = sorted(set(partition.values()))
    comm_to_idx = {c: i for i, c in enumerate(comm_list)}

    # Create new adjacency matrix
    from scipy.sparse import lil_matrix
    new_adj = lil_matrix((n_communities, n_communities))

    # Sum edge weights between communities
    rows, cols = adjacency_matrix.nonzero()
    for i, j in zip(rows, cols):
        comm_i = comm_to_idx[partition[i]]
        comm_j = comm_to_idx[partition[j]]

        if comm_i != comm_j:  # Only inter-clustering edges
            new_adj[comm_i, comm_j] += adjacency_matrix[i, j]

    return new_adj.tocsr()
