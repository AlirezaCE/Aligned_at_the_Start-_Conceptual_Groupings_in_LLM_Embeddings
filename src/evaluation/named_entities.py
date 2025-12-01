"""
Named Entity Evaluator.

Evaluates detected clusters against external name and location databases.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter
import logging

from ..data.datasets import DatasetManager

logger = logging.getLogger(__name__)


class NamedEntityEvaluator:
    """
    Evaluate conceptual clusters using external name and location datasets.

    Calculates precision scores by matching cluster tokens to:
    - Human names (by country, gender)
    - Geographic locations (by country, region)
    """

    def __init__(
        self,
        data_dir: str = "./data",
        min_cluster_size: int = 10,
        category_threshold: float = 0.7
    ):
        """
        Initialize the evaluator.

        Args:
            data_dir: Directory with datasets
            min_cluster_size: Minimum cluster size to evaluate
            category_threshold: Minimum ratio to assign a category
        """
        self.data_dir = data_dir
        self.min_cluster_size = min_cluster_size
        self.category_threshold = category_threshold

        # Load datasets
        logger.info("Loading external datasets...")
        self.dataset_manager = DatasetManager(data_dir)

        self.name_data = None
        self.location_data = None

        try:
            self.name_data = self.dataset_manager.load_name_dataset()
        except Exception as e:
            logger.warning(f"Could not load name dataset: {e}")

        try:
            self.location_data = self.dataset_manager.load_location_dataset()
        except Exception as e:
            logger.warning(f"Could not load location dataset: {e}")

    def evaluate_cluster(self, tokens: List[str]) -> Dict:
        """
        Evaluate a single cluster against external datasets.

        Args:
            tokens: List of token strings in the cluster

        Returns:
            Dictionary with evaluation results
        """
        results = {
            'size': len(tokens),
            'name_match': None,
            'location_match': None,
            'category': 'unknown',
            'precision': 0.0,
            'support': 0
        }

        # Try name matching
        if self.name_data is not None:
            name_results = self._evaluate_names(tokens)
            if name_results['precision'] > results['precision']:
                results.update(name_results)

        # Try location matching
        if self.location_data is not None:
            loc_results = self._evaluate_locations(tokens)
            if loc_results['precision'] > results['precision']:
                results.update(loc_results)

        return results

    def _evaluate_names(self, tokens: List[str]) -> Dict:
        """
        Evaluate tokens against name dataset.

        Args:
            tokens: List of token strings

        Returns:
            Dictionary with name matching results
        """
        if 'first_names' not in self.name_data and 'last_names' not in self.name_data:
            return {'precision': 0.0, 'category': 'unknown', 'support': 0}

        # Check both first and last names
        first_matches = []
        last_matches = []

        if 'first_names' in self.name_data:
            first_df = self.name_data['first_names']
            first_name_set = set(first_df['name'].str.lower()) if 'name' in first_df.columns else set()
            first_matches = [t for t in tokens if t.lower() in first_name_set]

        if 'last_names' in self.name_data:
            last_df = self.name_data['last_names']
            last_name_set = set(last_df['name'].str.lower()) if 'name' in last_df.columns else set()
            last_matches = [t for t in tokens if t.lower() in last_name_set]

        # Determine if this is primarily a first name or last name cluster
        first_ratio = len(first_matches) / len(tokens) if tokens else 0
        last_ratio = len(last_matches) / len(tokens) if tokens else 0

        if first_ratio > last_ratio and first_ratio > 0.5:
            category_type = "first"
            matches = first_matches
            precision = first_ratio
        elif last_ratio > first_ratio and last_ratio > 0.5:
            category_type = "last"
            matches = last_matches
            precision = last_ratio
        else:
            # Not a clear name cluster
            return {'precision': 0.0, 'category': 'unknown', 'support': 0}

        # Try to determine country/gender if first names
        category = f"{category_type}_names"
        if category_type == "first" and 'first_names' in self.name_data:
            # Simplified: just report as names without detailed country analysis
            # (Full implementation would analyze country distributions)
            category = "names"

        return {
            'precision': precision,
            'category': category,
            'support': len(matches),
            'name_match': True
        }

    def _evaluate_locations(self, tokens: List[str]) -> Dict:
        """
        Evaluate tokens against location dataset.

        Args:
            tokens: List of token strings

        Returns:
            Dictionary with location matching results
        """
        if not self.location_data:
            return {'precision': 0.0, 'category': 'unknown', 'support': 0}

        # Build location name sets
        location_names = set()

        if 'countries' in self.location_data:
            for country in self.location_data['countries']:
                if 'name' in country:
                    location_names.add(country['name'].lower())

        if 'states' in self.location_data:
            for state in self.location_data['states']:
                if 'name' in state:
                    location_names.add(state['name'].lower())

        if 'cities' in self.location_data:
            for city in self.location_data['cities']:
                if 'name' in city:
                    location_names.add(city['name'].lower())

        # Match tokens
        matches = [t for t in tokens if t.lower() in location_names]
        precision = len(matches) / len(tokens) if tokens else 0

        if precision > 0.5:
            return {
                'precision': precision,
                'category': 'locations',
                'support': len(matches),
                'location_match': True
            }

        return {'precision': 0.0, 'category': 'unknown', 'support': 0}

    def evaluate_all_clusters(self, communities: List[Dict]) -> pd.DataFrame:
        """
        Evaluate all detected communities.

        Args:
            communities: List of clustering dictionaries with 'tokens' field

        Returns:
            DataFrame with evaluation results
        """
        logger.info(f"Evaluating {len(communities)} clusters...")

        results = []
        for comm in communities:
            if len(comm['tokens']) < self.min_cluster_size:
                continue

            eval_results = self.evaluate_cluster(comm['tokens'])

            # Add cluster metadata
            results.append({
                'cluster_id': comm.get('id', -1),
                'k': comm.get('k', -1),
                'size': len(comm['tokens']),
                'category': eval_results['category'],
                'precision': eval_results['precision'],
                'support': eval_results['support'],
                'sample_tokens': ', '.join(comm['tokens'][:5])  # First 5 tokens
            })

        df = pd.DataFrame(results)

        # Log summary
        if len(df) > 0:
            logger.info(f"\nEvaluation Summary:")
            logger.info(f"  Total clusters evaluated: {len(df)}")
            logger.info(f"  Categories found: {df['category'].value_counts().to_dict()}")
            logger.info(f"  Mean precision: {df['precision'].mean():.3f}")
            logger.info(f"  High precision clusters (>0.7): {(df['precision'] > 0.7).sum()}")

        return df

    def get_high_precision_clusters(
        self,
        df: pd.DataFrame,
        min_precision: float = 0.7
    ) -> pd.DataFrame:
        """
        Filter clusters with high precision.

        Args:
            df: DataFrame from evaluate_all_clusters
            min_precision: Minimum precision threshold

        Returns:
            Filtered DataFrame
        """
        return df[df['precision'] >= min_precision].sort_values('precision', ascending=False)

    def generate_report(self, df: pd.DataFrame, output_path: Optional[str] = None) -> str:
        """
        Generate a text report of evaluation results.

        Args:
            df: DataFrame from evaluate_all_clusters
            output_path: Optional path to save report

        Returns:
            Report text
        """
        report_lines = [
            "=" * 80,
            "Named Entity Evaluation Report",
            "=" * 80,
            f"\nTotal Clusters Evaluated: {len(df)}",
            f"Mean Precision: {df['precision'].mean():.3f}",
            f"Median Precision: {df['precision'].median():.3f}",
            "\nCategory Distribution:"
        ]

        # Category counts
        for category, count in df['category'].value_counts().items():
            mean_prec = df[df['category'] == category]['precision'].mean()
            report_lines.append(f"  {category}: {count} clusters (avg precision: {mean_prec:.3f})")

        # High precision clusters
        high_prec = df[df['precision'] > 0.7]
        report_lines.append(f"\nHigh Precision Clusters (>0.7): {len(high_prec)}")

        for idx, row in high_prec.head(10).iterrows():
            report_lines.append(
                f"  Cluster {row['cluster_id']}: {row['category']} "
                f"(precision={row['precision']:.3f}, size={row['size']})"
            )
            report_lines.append(f"    Sample: {row['sample_tokens']}")

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")

        return report
