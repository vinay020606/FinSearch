"""
Concrete Embedding Providers for Financial RAG.
Supports BGE-large-en-v1.5, Cohere embed-v3, e5-mistral-7b-instruct, Finance-domain models, and Fallback mode.
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


class BGELargeEnEmbedding(BaseEmbeddingModel):
    """
    BAAI/bge-large-en-v1.5 Embedding Provider.
    Open-source, deployable locally via sentence-transformers.
    Applies required query instruction prefix: 'Represent this sentence for searching relevant passages: '
    """

    def __init__(self, model_name_or_path: str = "BAAI/bge-large-en-v1.5"):
        self._model_name = model_name_or_path
        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        except ImportError:
            warnings.warn("sentence-transformers is not installed. Falling back to FallbackHashEmbedding logic.")
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
        # BAAI BGE model requirement for query encoding
        instruction_query = f"Represent this sentence for searching relevant passages: {query}"
        if self._model:
            embedding = self._model.encode(instruction_query, normalize_embeddings=True)
            return embedding.tolist()
        return self._fallback.embed_query(instruction_query)


class CohereEmbedV3(BaseEmbeddingModel):
    """
    Cohere embed-v3 Commercial API Provider (embed-english-v3.0 / embed-multilingual-v3.0).
    Uses input_type='search_document' for chunks and input_type='search_query' for queries.
    """

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
            except ImportError:
                warnings.warn("cohere SDK not installed. Falling back to FallbackHashEmbedding.")
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
            res = self._client.embed(
                texts=texts,
                model=self._model_name,
                input_type="search_document"
            )
            return res.embeddings
        return self._fallback.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        if self._client:
            res = self._client.embed(
                texts=[query],
                model=self._model_name,
                input_type="search_query"
            )
            return res.embeddings[0]
        return self._fallback.embed_query(query)


class E5Mistral7BEmbedding(BaseEmbeddingModel):
    """
    intfloat/e5-mistral-7b-instruct Embedding Provider.
    Instruction-tuned asymmetric retrieval model (4096 dimensions).
    Applies custom instruction prompt for financial search tasks.
    """

    def __init__(self, model_name_or_path: str = "intfloat/e5-mistral-7b-instruct"):
        self._model_name = model_name_or_path
        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        except ImportError:
            warnings.warn("sentence-transformers not installed. Falling back to FallbackHashEmbedding.")
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
        # E5-Mistral custom instruction format
        task_instruction = "Instruct: Given a financial query, retrieve relevant section context and tables from financial reports.\nQuery: "
        instruct_query = f"{task_instruction}{query}"
        if self._model:
            embedding = self._model.encode(instruct_query, normalize_embeddings=True)
            return embedding.tolist()
        return self._fallback.embed_query(instruct_query)


class FinanceDomainAdaptedEmbedding(BaseEmbeddingModel):
    """
    Finance-Domain Adapted Model Provider (e.g., FinBGE / FinBERT / Salesforce SFR-Embedding-Mistral).
    Optimized for specialized financial terminology, accounting standard codes, and balance sheet metrics.
    """

    def __init__(self, model_name_or_path: str = "BAAI/bge-large-en-v1.5-finetuned-finance"):
        self._model_name = model_name_or_path
        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        except ImportError:
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
        if self._model:
            embedding = self._model.encode(query, normalize_embeddings=True)
            return embedding.tolist()
        return self._fallback.embed_query(query)
