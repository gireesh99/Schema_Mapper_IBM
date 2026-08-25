"""Generate embeddings and search for similar fields using FAISS with BM25 hybrid scoring."""

import numpy as np
import faiss
from openai import OpenAI
from rank_bm25 import BM25Okapi
import logging

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Create embeddings and search for similar fields using BM25 + semantic hybrid search."""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = "text-embedding-3-small"
        self.cache = {}
        self.index = None
        self.field_map = {}
        self.bm25_index = None
        self.bm25_corpus = []

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from cache or OpenAI."""
        if text in self.cache:
            return self.cache[text]

        logger.info(f"Embedding: {text[:50]}...")
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        embedding = np.array(response.data[0].embedding, dtype="float32")
        self.cache[text] = embedding
        return embedding

    def build_index(self, fields) -> None:
        """Build FAISS semantic index and BM25 keyword index."""
        logger.info(f"Building indices for {len(fields)} fields...")

        embeddings = []
        self.bm25_corpus = []

        for i, field in enumerate(fields):
            # Semantic index
            emb = self.get_embedding(field.get_search_text())
            embeddings.append(emb)
            self.field_map[i] = field

            # BM25 index - tokenize field name and description
            tokens = (field.name + " " + field.description).lower().split()
            self.bm25_corpus.append(tokens)

        # Build FAISS semantic index
        embeddings_array = np.array(embeddings).astype("float32")
        self.index = faiss.IndexFlatL2(embeddings_array.shape[1])
        self.index.add(embeddings_array)

        # Build BM25 keyword index
        self.bm25_index = BM25Okapi(self.bm25_corpus)

        logger.info(f"Semantic + BM25 indices ready with {self.index.ntotal} fields")

    def search_similar_fields(self, query_field, k: int = 1):
        """Find k most similar fields using BM25 + semantic hybrid search."""
        # 1. Semantic search with FAISS
        emb = self.get_embedding(query_field.get_search_text())
        query_array = np.array([emb]).astype("float32")
        distances, indices = self.index.search(query_array, k * 3)

        # 2. BM25 keyword search
        query_tokens = (query_field.name + " " + query_field.description).lower().split()
        bm25_scores = self.bm25_index.get_scores(query_tokens)

        # 3. Score each candidate with hybrid approach
        scored_results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                target_field = self.field_map[idx]

                # Semantic score (embedding similarity)
                semantic_score = 1 / (1 + float(dist))

                # Keyword score (BM25 relevance)
                # Normalize BM25 score to 0-1 range
                bm25_score = min(bm25_scores[idx] / 10.0, 1.0)

                # Hybrid combination (70% semantic, 30% BM25)
                # Higher semantic weight because BM25 struggles with short names
                hybrid_score = semantic_score * 0.7 + bm25_score * 0.3

                scored_results.append((target_field, hybrid_score))

        # Sort by hybrid score and return top k
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [(field, score) for field, score in scored_results[:k]]
