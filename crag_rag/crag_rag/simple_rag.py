from .document_encoder import DocumentEncoder
from .vector_store import VectorStore
from .llm_generator import LLMGenerator
from typing import List

class SimpleRAG:
    def __init__(self, document_encoder: DocumentEncoder, vector_store: VectorStore, llm_generator: LLMGenerator):
        self.document_encoder = document_encoder
        self.vector_store = vector_store
        self.llm_generator = llm_generator
        print("SimpleRAG system initialized.")

    def answer_question(self, query_text: str, k: int = 5) -> str:
        query_embedding = self.document_encoder.encode([query_text])[0]
        retrieved_results = self.vector_store.search(query_embedding, k=k)
        contexts = [result[2]['text'] for result in retrieved_results]
        if not contexts:
            return "No relevant documents found."
        answer = self.llm_generator.generate_answer(question=query_text, context=contexts)
        return answer
