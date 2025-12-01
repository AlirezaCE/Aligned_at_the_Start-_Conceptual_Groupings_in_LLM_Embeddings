"""
Fuzzy graph construction using UMAP-based weighting.

Implements the fuzzy simplex weighting from Section 3.1 of the paper.
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial.distance import cosine
from scipy.optimize import root_scalar
from sklearn.neighbors import NearestNeighbors
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class FuzzyGraphBuilder:
    """
    Build a fuzzy k-NN graph with UMAP-based edge weighting.

    The weighting formula is:
        ω(xi, xj) = exp(−max(0, d(xi, xj) − ρi) / σi)

    where:
    - d(xi, xj) is the cosine distance
    - ρi is the minimum distance to k-nearest neighbors
    - σi is calculated such that Σω(xi, xj) = log2(k)
    """

    def __init__(self, embeddings: np.ndarray, k: int, metric: str = "cosine"):
        """
        Initialize the fuzzy graph builder.

        Args:
            embeddings: Array of shape (n_samples, n_features)
            k: Number of nearest neighbors
            metric: Distance metric to use ("cosine" recommended)
        """
        self.embeddings = embeddings
        self.k = k
        self.metric = metric
        self.n_samples = embeddings.shape[0]

        logger.info(f"Initializing FuzzyGraphBuilder with {self.n_samples} samples, k={k}")

    def build_knn_graph(self) -> csr_matrix:
        """
        Build the fuzzy k-NN graph.

        Returns:
            Sparse adjacency matrix of shape (n_samples, n_samples)
        """
        logger.info("Finding k-nearest neighbors...")
        neighbors, distances = self._find_knn()

        logger.info("Computing rho values...")
        rho = self._compute_rho(distances)

        logger.info("Computing sigma values...")
        sigma = self._compute_sigma(distances, rho)

        logger.info("Computing edge weights...")
        adjacency = self._compute_weights(neighbors, distances, rho, sigma)

        return adjacency

    def _find_knn(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k-nearest neighbors for each embedding.

        Returns:
            neighbors: Array of shape (n_samples, k) with neighbor indices
            distances: Array of shape (n_samples, k) with distances
        """
        # Use sklearn's NearestNeighbors for efficiency
        # k+1 because the first neighbor is the point itself
        nbrs = NearestNeighbors(n_neighbors=self.k + 1, metric=self.metric, n_jobs=-1)
        nbrs.fit(self.embeddings)

        distances, neighbors = nbrs.kneighbors(self.embeddings)

        # Remove the first neighbor (self) and first distance (0)
        return neighbors[:, 1:], distances[:, 1:]

    def _compute_rho(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute rho_i for each node.

        ρi = min{d(xi, xj) | 1 ≤ j ≤ k, d(xi, xj) > 0}

        Args:
            distances: Array of shape (n_samples, k)

        Returns:
            Array of shape (n_samples,) with rho values
        """
        # Get minimum positive distance for each node
        # distances are already sorted from k-NN
        rho = distances[:, 0]  # First distance is the minimum

        # Handle case where minimum distance is 0 (duplicate embeddings)
        # In this case, use the second minimum
        zero_mask = rho == 0
        if np.any(zero_mask):
            # For nodes with zero distance, find first non-zero distance
            for i in np.where(zero_mask)[0]:
                non_zero = distances[i][distances[i] > 0]
                rho[i] = non_zero[0] if len(non_zero) > 0 else 1e-10

        return rho

    def _compute_sigma(self, distances: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """
        Compute sigma_i for each node such that Σω = log2(k).

        Solves the equation:
            Σ exp(−max(0, d(xi, xj) − ρi) / σi) = log2(k)

        Args:
            distances: Array of shape (n_samples, k)
            rho: Array of shape (n_samples,)

        Returns:
            Array of shape (n_samples,) with sigma values
        """
        target = np.log2(self.k)
        sigma = np.zeros(self.n_samples)

        for i in range(self.n_samples):
            d_i = distances[i]
            rho_i = rho[i]

            # Define the function to solve: sum of weights - target = 0
            def weight_sum(sig):
                if sig <= 0:
                    return float('inf')
                weights = np.exp(-np.maximum(0, d_i - rho_i) / sig)
                return np.sum(weights) - target

            # Binary search for sigma
            # Start with initial guess
            sigma_guess = np.mean(np.maximum(0, d_i - rho_i)) + 1e-10

            try:
                # Try to find root using binary search
                result = root_scalar(
                    weight_sum,
                    bracket=[1e-10, 100.0],
                    method='brentq',
                    xtol=1e-6
                )
                sigma[i] = result.root
            except (ValueError, RuntimeError):
                # If root finding fails, use simple approximation
                sigma[i] = sigma_guess

        return sigma

    def _compute_weights(
        self,
        neighbors: np.ndarray,
        distances: np.ndarray,
        rho: np.ndarray,
        sigma: np.ndarray
    ) -> csr_matrix:
        """
        Compute edge weights using the fuzzy simplex formula.

        Args:
            neighbors: Array of shape (n_samples, k)
            distances: Array of shape (n_samples, k)
            rho: Array of shape (n_samples,)
            sigma: Array of shape (n_samples,)

        Returns:
            Sparse weighted adjacency matrix
        """
        # Use lil_matrix for efficient construction
        adjacency = lil_matrix((self.n_samples, self.n_samples))

        for i in range(self.n_samples):
            for j_idx, j in enumerate(neighbors[i]):
                d_ij = distances[i, j_idx]
                rho_i = rho[i]
                sigma_i = sigma[i]

                # Compute weight: ω(xi, xj) = exp(-max(0, d - ρ) / σ)
                weight = np.exp(-max(0, d_ij - rho_i) / sigma_i)
                adjacency[i, j] = weight

        # Convert to CSR for efficient operations
        return adjacency.tocsr()


def validate_graph_weights(adjacency: csr_matrix, k: int, tolerance: float = 0.1) -> bool:
    """
    Validate that row sums approximately equal log2(k).

    Args:
        adjacency: Sparse adjacency matrix
        k: Number of neighbors
        tolerance: Allowed relative error

    Returns:
        True if validation passes
    """
    target = np.log2(k)
    row_sums = np.array(adjacency.sum(axis=1)).flatten()

    # Check if row sums are close to target
    errors = np.abs(row_sums - target) / target
    max_error = np.max(errors)
    mean_error = np.mean(errors)

    logger.info(f"Graph validation: target={target:.3f}, max_error={max_error:.3f}, mean_error={mean_error:.3f}")

    return max_error < tolerance
