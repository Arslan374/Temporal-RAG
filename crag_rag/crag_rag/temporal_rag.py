import re
from typing import Dict, List
from .time_aware import TimeAwareModule
from .query_router import QueryRouter
from .vector_store import VectorStore
from .llm_generator import LLMGenerator

class TemporalRAGPipeline:
    def __init__(self, time_aware_module: TimeAwareModule, 
                 query_router: QueryRouter, 
                 vector_store: VectorStore, 
                 llm_generator: LLMGenerator,
                 re_ranking_weights: Dict[str, float] = None):
        self.time_aware_module = time_aware_module
        self.query_router = query_router
        self.vector_store = vector_store
        self.llm_generator = llm_generator
        self.re_ranking_weights = re_ranking_weights if re_ranking_weights else {
            'semantic': 0.5,
            'temporal_metadata': 0.3,
            'temporal_content': 0.2
        }
        total_weight = sum(self.re_ranking_weights.values())
        if total_weight > 0:
            self.re_ranking_weights = {k: v / total_weight for k, v in self.re_ranking_weights.items()}
        print(f"TemporalRAGPipeline initialized with re-ranking weights: {self.re_ranking_weights}")

    def answer_question(self, query_text: str, k_retrieve: int = 10, k_rerank: int = 5) -> str:
        query_embedding = self.query_router.route_and_encode(query_text)
        retrieved_results = self.vector_store.search(query_embedding, k=k_retrieve)
        if not retrieved_results:
            return "No documents retrieved for re-ranking."
        scored_documents = []
        for doc_id, l2_distance, metadata in retrieved_results:
            doc_text = metadata['text']
            doc_timestamp = metadata['timestamp']
            semantic_score = 1.0 / (1.0 + l2_distance) if l2_distance >= 0 else 0.0
            query_timestamp = None
            if self.time_aware_module.is_temporal_query(query_text):
                year_match = re.search(r'\b\d{4}\b', query_text)
                if year_match:
                    query_timestamp = year_match.group(0)
            temporal_metadata_score = self.time_aware_module.get_temporal_relevance_from_timestamps(
                query_timestamp, doc_timestamp
            )
            temporal_content_score = self.time_aware_module.get_temporal_score(doc_text)
            final_re_ranking_score = (
                self.re_ranking_weights.get('semantic', 0) * semantic_score +
                self.re_ranking_weights.get('temporal_metadata', 0) * temporal_metadata_score +
                self.re_ranking_weights.get('temporal_content', 0) * temporal_content_score
            )
            scored_documents.append({'id': doc_id, 'text': doc_text, 'score': final_re_ranking_score})
        scored_documents.sort(key=lambda x: x['score'], reverse=True)
        top_reranked_contexts = [doc['text'] for doc in scored_documents[:k_rerank]]
        if not top_reranked_contexts:
            return "No documents left after re-ranking for answer generation."
        answer = self.llm_generator.generate_answer(question=query_text, context=top_reranked_contexts)
        return answer
