"""
Dataset download and loading utilities.

Downloads and processes external datasets for evaluation:
- name-dataset: Human names by country and gender
- country-state-city: Geographic locations
"""

import os
import json
import pandas as pd
import requests
from pathlib import Path
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import zipfile
import shutil

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Manage external datasets for evaluation.
    """

    def __init__(self, data_dir: str = "./data"):
        """
        Initialize dataset manager.

        Args:
            data_dir: Root directory for datasets
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Dataset manager initialized at: {self.data_dir}")

    def download_name_dataset(self, force: bool = False) -> Path:
        """
        Download name-dataset from GitHub.

        Args:
            force: Force re-download even if exists

        Returns:
            Path to downloaded data
        """
        output_dir = self.raw_dir / "name-dataset"

        if output_dir.exists() and not force:
            logger.info(f"Name dataset already exists at: {output_dir}")
            return output_dir

        logger.info("Downloading name-dataset...")

        # GitHub raw URLs for the pickle files in v3
        base_url = "https://raw.githubusercontent.com/philipperemy/name-dataset/master/names_dataset/v3/"

        files_to_download = {
            "first_names.pkl.gz": "first_names.pkl.gz",
            "last_names.pkl.gz": "last_names.pkl.gz"
        }

        output_dir.mkdir(parents=True, exist_ok=True)

        for filename, url_path in files_to_download.items():
            url = base_url + url_path
            output_path = output_dir / filename

            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logger.info(f"Downloaded: {filename}")
            except Exception as e:
                logger.warning(f"Could not download {filename}: {e}")

        return output_dir

    def download_location_dataset(self, force: bool = False) -> Path:
        """
        Download country-state-city database.

        Args:
            force: Force re-download even if exists

        Returns:
            Path to downloaded data
        """
        output_dir = self.raw_dir / "countries-states-cities"

        if output_dir.exists() and not force:
            logger.info(f"Location dataset already exists at: {output_dir}")
            return output_dir

        logger.info("Downloading countries-states-cities database...")

        # GitHub raw URL
        base_url = "https://raw.githubusercontent.com/dr5hn/countries-states-cities-database/master/json/"

        # cities.json is compressed, so download cities.json.gz
        files = ["countries.json", "states.json", "cities.json.gz"]

        output_dir.mkdir(parents=True, exist_ok=True)

        for filename in files:
            url = base_url + filename

            # For .gz files, save compressed then decompress
            if filename.endswith('.gz'):
                output_path_gz = output_dir / filename
                output_path = output_dir / filename.replace('.gz', '')

                try:
                    response = requests.get(url)
                    response.raise_for_status()

                    # Save compressed file
                    with open(output_path_gz, 'wb') as f:
                        f.write(response.content)

                    # Decompress
                    import gzip
                    with gzip.open(output_path_gz, 'rb') as f_in:
                        with open(output_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    # Remove compressed file
                    output_path_gz.unlink()

                    logger.info(f"Downloaded and decompressed: {filename}")
                except Exception as e:
                    logger.warning(f"Could not download {filename}: {e}")
            else:
                output_path = output_dir / filename
                try:
                    response = requests.get(url)
                    response.raise_for_status()

                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(response.json(), f, indent=2)

                    logger.info(f"Downloaded: {filename}")
                except Exception as e:
                    logger.warning(f"Could not download {filename}: {e}")

        return output_dir

    def load_name_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Load name dataset into DataFrames.

        Returns:
            Dictionary with 'first_names' and 'last_names' DataFrames
        """
        import pickle
        import gzip

        name_dir = self.raw_dir / "name-dataset"

        if not name_dir.exists():
            logger.info("Name dataset not found, downloading...")
            self.download_name_dataset()

        data = {}

        # Load first names from pickle
        first_names_path = name_dir / "first_names.pkl.gz"
        if first_names_path.exists():
            try:
                with gzip.open(first_names_path, 'rb') as f:
                    first_names_dict = pickle.load(f)
                # Convert dict to DataFrame with 'name' column
                if isinstance(first_names_dict, dict):
                    data['first_names'] = pd.DataFrame({'name': list(first_names_dict.keys())})
                elif isinstance(first_names_dict, list):
                    data['first_names'] = pd.DataFrame({'name': first_names_dict})
                else:
                    data['first_names'] = first_names_dict
                logger.info(f"Loaded {len(data['first_names'])} first names")
            except Exception as e:
                logger.warning(f"Could not load first names: {e}")

        # Load last names from pickle
        last_names_path = name_dir / "last_names.pkl.gz"
        if last_names_path.exists():
            try:
                with gzip.open(last_names_path, 'rb') as f:
                    last_names_dict = pickle.load(f)
                # Convert dict to DataFrame with 'name' column
                if isinstance(last_names_dict, dict):
                    data['last_names'] = pd.DataFrame({'name': list(last_names_dict.keys())})
                elif isinstance(last_names_dict, list):
                    data['last_names'] = pd.DataFrame({'name': last_names_dict})
                else:
                    data['last_names'] = last_names_dict
                logger.info(f"Loaded {len(data['last_names'])} last names")
            except Exception as e:
                logger.warning(f"Could not load last names: {e}")

        return data

    def load_location_dataset(self) -> Dict[str, List[Dict]]:
        """
        Load location dataset.

        Returns:
            Dictionary with 'countries', 'states', 'cities' lists
        """
        location_dir = self.raw_dir / "countries-states-cities"

        if not location_dir.exists():
            logger.info("Location dataset not found, downloading...")
            self.download_location_dataset()

        data = {}

        for filename in ['countries', 'states', 'cities']:
            filepath = location_dir / f"{filename}.json"
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data[filename] = json.load(f)
                logger.info(f"Loaded {len(data[filename])} {filename}")

        return data

    def setup_all_datasets(self, force: bool = False):
        """
        Download all required datasets.

        Args:
            force: Force re-download
        """
        logger.info("Setting up all datasets...")
        self.download_name_dataset(force=force)
        self.download_location_dataset(force=force)
        logger.info("All datasets ready!")


# Convenience function
def setup_datasets(data_dir: str = "./data", force: bool = False):
    """
    Setup all required datasets.

    Args:
        data_dir: Root directory for datasets
        force: Force re-download
    """
    manager = DatasetManager(data_dir)
    manager.setup_all_datasets(force=force)
