"""
Base Chunking Strategy interface.
"""

from abc import ABC, abstractmethod
from typing import List
from ..types import Document, Chunk


class BaseChunkingStrategy(ABC):
    """Abstract Base Class for Document-Type Specific Chunking Strategies."""

    @abstractmethod
    def chunk(self, doc: Document) -> List[Chunk]:
        """
        Processes a document into structured, hierarchical, contextual chunks.
        
        Args:
            doc: Input Document containing raw text, metadata, and document type.
            
        Returns:
            List of Chunk objects containing parent sections and leaf child chunks.
        """
        pass
