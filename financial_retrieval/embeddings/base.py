"""
Abstract Base Class for Financial RAG Embedding Providers.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingModel(ABC):
    """Abstract Base Class for dense vector embedding models in Financial RAG."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the embedding vector dimension size."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the model identifier name."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of document chunk texts for vector indexing.
        
        Args:
            texts: List of chunk text strings (including contextual prefixes).
            
        Returns:
            List of normalized floating point embedding vectors.
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a search query string for retrieval.
        Applies model-specific query prefixes or instruction prompts if required.
        
        Args:
            query: User search query string.
            
        Returns:
            Normalized floating point query embedding vector.
        """
        pass
