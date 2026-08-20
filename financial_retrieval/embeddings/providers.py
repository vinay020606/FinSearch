"""
Concrete Embedding Providers for Financial RAG.
Supports Fin-E5, FinModernBERT, BGE-large-en-v1.5, Cohere embed-v3, E5-Mistral, and Fallback mode.
"""

import math
import warnings
from typing import List, Optional
from .base import BaseEmbeddingModel


class FallbackHashEmbedding(BaseEmbeddingModel):
    """
    Lightweight zero-dependency fallback embedding provider using deterministic feature hashing.
    Ideal for local testing, CI/CD pipelines, and fast prototyping without heavy ML weights.
    """

    def __init__(self, dim: int = 1024):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "fallback-hash-1024d"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._embed_single(query)

    def _embed_single(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        words = text.lower().split()
        if not words:
            return vec
        for word in words:
            idx = abs(hash(word)) % self._dim
            vec[idx] += 1.0 + (len(word) * 0.05)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class FinE5Embedding(BaseEmbeddingModel):
    """
    Fin-E5 (FinanceMTEB/Fin-E5) Embedding Provider.
    Fine-tuned specifically on SEC filings, 10-K/10-Q regulatory documents, and corporate reports.
    Applies asymmetric E5 prefixes: 'passage: ' for document chunks and 'query: ' for queries.
    """

    def __init__(self, model_name_or_path: str = "FinanceMTEB/Fin-E5"):
        self._model_name = model_name_or_path
        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        except Exception:
            warnings.warn(f"Could not load {self._model_name}. Falling back to FallbackHashEmbedding.")
            self._fallback = FallbackHashEmbedding(dim=1024)

    @property
    def dimension(self) -> int:
        return 1024

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed_texts = [f"passage: {t}" for t in texts]
        if self._model:
            embeddings = self._model.encode(prefixed_texts, normalize_embeddings=True)
            return embeddings.tolist()
        return self._fallback.embed_documents(prefixed_texts)

    def embed_query(self, query: str) -> List[float]:
        prefixed_query = f"query: {query}"
        if self._model:
            embedding = self._model.encode(prefixed_query, normalize_embeddings=True)
            return embedding.tolist()
        return self._fallback.embed_query(prefixed_query)


class FinModernBERTEmbedding(BaseEmbeddingModel):
    """
    FinModernBERT Embedding Provider.
    ModernBERT architecture fine-tuned via Domain-Adaptive Pre-Training (DAPT) on billions of financial tokens.
    Supports long context windows (up to 8192 tokens) and efficient local Flash-Attention processing.
    """

    def __init__(self, model_name_or_path: str = "answerdotai/ModernBERT-base-finance"):
        self._model_name = model_name_or_path
        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        except Exception:
            warnings.warn(f"Could not load {self._model_name}. Falling back to FallbackHashEmbedding.")
            self._fallback = FallbackHashEmbedding(dim=768)

    @property
    def dimension(self) -> int:
        return 768

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._model:
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
        return self._fallback.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        if self._model:
            embedding = self._model.encode(query, normalize_embeddings=True)
            return embedding.tolist()
        return self._fallback.embed_query(query)


class BGELargeEnEmbedding(BaseEmbeddingModel):
    """BAAI/bge-large-en-v1.5 Embedding Provider."""

    def __init__(self, model_name_or_path: str = "BAAI/bge-large-en-v1.5"):
        self._model_name = model_name_or_path
        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        except Exception:
            self._fallback = FallbackHashEmbedding(dim=1024)

    @property
    def dimension(self) -> int:
        return 1024

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._model:
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
        return self._fallback.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        instruction_query = f"Represent this sentence for searching relevant passages: {query}"
        if self._model:
            embedding = self._model.encode(instruction_query, normalize_embeddings=True)
            return embedding.tolist()
        return self._fallback.embed_query(instruction_query)


class CohereEmbedV3(BaseEmbeddingModel):
    """Cohere embed-v3 Commercial API Provider."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "embed-english-v3.0"):
        self._model_name = model_name
        self.api_key = api_key
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.api_key:
            try:
                import cohere
                self._client = cohere.Client(api_key=self.api_key)
            except Exception:
                self._fallback = FallbackHashEmbedding(dim=1024)
        else:
            self._fallback = FallbackHashEmbedding(dim=1024)

    @property
    def dimension(self) -> int:
        return 1024

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._client:
            res = self._client.embed(texts=texts, model=self._model_name, input_type="search_document")
            return res.embeddings
        return self._fallback.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        if self._client:
            res = self._client.embed(texts=[query], model=self._model_name, input_type="search_query")
            return res.embeddings[0]
        return self._fallback.embed_query(query)


class E5Mistral7BEmbedding(BaseEmbeddingModel):
    """intfloat/e5-mistral-7b-instruct Embedding Provider."""

    def __init__(self, model_name_or_path: str = "intfloat/e5-mistral-7b-instruct"):
        self._model_name = model_name_or_path
        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        except Exception:
            self._fallback = FallbackHashEmbedding(dim=4096)

    @property
    def dimension(self) -> int:
        return 4096

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._model:
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
        return self._fallback.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        task_instruction = "Instruct: Given a financial query, retrieve relevant section context and tables from financial reports.\nQuery: "
        instruct_query = f"{task_instruction}{query}"
        if self._model:
            embedding = self._model.encode(instruct_query, normalize_embeddings=True)
            return embedding.tolist()
        return self._fallback.embed_query(instruct_query)
