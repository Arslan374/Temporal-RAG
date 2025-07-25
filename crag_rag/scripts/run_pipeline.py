import os
import sys
import json
from crag_rag.data_utils import preprocess_crag_sample
from crag_rag.document_encoder import DocumentEncoder
from crag_rag.vector_store import VectorStore
from crag_rag.time_aware import TimeAwareModule
from crag_rag.query_encoder import NormalQueryEncoder, TemporalQueryEncoder
from crag_rag.query_router import QueryRouter
from crag_rag.llm_generator import LLMGenerator
from crag_rag.temporal_rag import TemporalRAGPipeline

def load_vector_store(index_dir: str) -> VectorStore:
    index_path = os.path.join(index_dir, 'document_index.faiss')
    vector_store = VectorStore(embedding_dim=384)  # Adjust if using a different encoder
    vector_store.load_index(index_path)
    return vector_store

def main(dataset_path: str, index_dir: str):
    # Load test questions
    with open(dataset_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    questions = [preprocess_crag_sample(s) for s in samples]
    # Load modules
    time_aware = TimeAwareModule()
    normal_enc = NormalQueryEncoder()
    temporal_enc = TemporalQueryEncoder(model_path='path/to/your/fine_tuned_contriever.bin')  # Update path
    router = QueryRouter(time_aware, normal_enc, temporal_enc)
    vector_store = load_vector_store(index_dir)
    llm = LLMGenerator()
    pipeline = TemporalRAGPipeline(time_aware, router, vector_store, llm)
    # Run pipeline for each question
    for i, q in enumerate(questions):
        print(f"\n--- Question {i+1} ---")
        print(f"Q: {q['text']}")
        answer = pipeline.answer_question(q['text'])
        print(f"A: {answer}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python run_pipeline.py <crag_dataset.jsonl> <index_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
