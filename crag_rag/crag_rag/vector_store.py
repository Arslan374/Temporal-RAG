import faiss
import numpy as np
import json
from typing import List, Dict, Tuple

class VectorStore:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.doc_ids = []
        self.metadata = []
        print(f"VectorStore initialized with embedding dimension: {embedding_dim}")

    def add_documents(self, embeddings: np.ndarray, doc_ids: List[str], metadata: List[Dict]):
        if embeddings.shape[0] != len(doc_ids) or embeddings.shape[0] != len(metadata):
            raise ValueError("Mismatched dimensions for embeddings, doc_ids, and metadata.")
        self.index.add(embeddings)
        self.doc_ids.extend(doc_ids)
        self.metadata.extend(metadata)
        print(f"Added {embeddings.shape[0]} documents to VectorStore.")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float, Dict]]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i == -1:
                continue
            doc_id = self.doc_ids[i]
            doc_meta = self.metadata[i]
            results.append((doc_id, dist, doc_meta))
        return results

    def save_index(self, path: str):
        faiss.write_index(self.index, path)
        with open(f"{path}.doc_ids.json", 'w') as f:
            json.dump(self.doc_ids, f)
        with open(f"{path}.metadata.json", 'w') as f:
            json.dump(self.metadata, f)
        print(f"VectorStore index saved to {path}")

    def load_index(self, path: str):
        self.index = faiss.read_index(path)
        with open(f"{path}.doc_ids.json", 'r') as f:
            self.doc_ids = json.load(f)
        with open(f"{path}.metadata.json", 'r') as f:
            self.metadata = json.load(f)
        print(f"VectorStore index loaded from {path}")
