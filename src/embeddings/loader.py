"""
Embedding extraction module for loading input embeddings from HuggingFace models.
"""

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class EmbeddingLoader:
    """
    Load input embeddings from HuggingFace transformer models.

    This class extracts the BASE (input) embeddings - the initial token representations
    before they pass through transformer blocks. These are different from contextual
    embeddings which are the outputs of transformer layers.

    Supported models:
    - Albert: Uses word_embeddings from albert.embeddings
    - T5: Uses shared embeddings
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize the embedding loader.

        Args:
            model_name: HuggingFace model name (e.g., "albert-base-v2", "t5-base")
            device: Device to load model on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device

        logger.info(f"Loading model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model.eval()  # Set to evaluation mode

        # Extract embedding matrix
        self.embedding_matrix = self._extract_embeddings()
        logger.info(f"Extracted embeddings shape: {self.embedding_matrix.shape}")

    def _extract_embeddings(self) -> np.ndarray:
        """
        Extract input embedding weights from the model.

        Returns:
            numpy array of shape (vocab_size, embedding_dim)
        """
        with torch.no_grad():
            if "albert" in self.model_name.lower():
                # Albert: model.embeddings.word_embeddings.weight
                embeddings = self.model.embeddings.word_embeddings.weight
            elif "t5" in self.model_name.lower():
                # T5: model.shared.weight (shared encoder-decoder embeddings)
                embeddings = self.model.shared.weight
            elif "bert" in self.model_name.lower():
                # BERT: model.embeddings.word_embeddings.weight
                embeddings = self.model.embeddings.word_embeddings.weight
            elif "gpt" in self.model_name.lower():
                # GPT: model.wte.weight
                embeddings = self.model.wte.weight
            else:
                raise ValueError(f"Unsupported model type: {self.model_name}")

            # Convert to numpy
            return embeddings.cpu().numpy()

    def get_embeddings(self) -> np.ndarray:
        """
        Get the embedding matrix.

        Returns:
            numpy array of shape (vocab_size, embedding_dim)
        """
        return self.embedding_matrix

    def get_vocabulary(self) -> List[str]:
        """
        Get the vocabulary tokens as strings.

        Returns:
            List of token strings
        """
        vocab = self.tokenizer.get_vocab()
        # Sort by token ID
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        return [token for token, idx in sorted_vocab]

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_matrix.shape[1]

    def get_token_embedding(self, token: str) -> np.ndarray:
        """
        Get embedding for a specific token.

        Args:
            token: Token string

        Returns:
            numpy array of shape (embedding_dim,)
        """
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        return self.embedding_matrix[token_id]

    def get_embedding_info(self) -> dict:
        """
        Get information about the loaded embeddings.

        Returns:
            Dictionary with model info
        """
        return {
            "model_name": self.model_name,
            "vocab_size": self.get_vocab_size(),
            "embedding_dim": self.get_embedding_dim(),
            "device": self.device,
            "embedding_shape": self.embedding_matrix.shape
        }


def load_multiple_models(model_names: List[str], device: str = "cpu") -> dict:
    """
    Load embeddings from multiple models.

    Args:
        model_names: List of HuggingFace model names
        device: Device to load models on

    Returns:
        Dictionary mapping model names to EmbeddingLoader objects
    """
    loaders = {}
    for name in model_names:
        try:
            loaders[name] = EmbeddingLoader(name, device)
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")

    return loaders
