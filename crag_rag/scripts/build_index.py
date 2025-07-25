import os
import sys
import json
import bz2
from crag_rag.data_utils import load_json_dataset, preprocess_crag_document, preprocess_crag_sample
from crag_rag.document_encoder import DocumentEncoder
from crag_rag.vector_store import VectorStore

def build_document_index(dataset_path: str, output_dir: str = 'index_data'):
    os.makedirs(output_dir, exist_ok=True)
    # Load and preprocess dataset
    if dataset_path.endswith('.bz2'):
        with bz2.open(dataset_path, 'rt', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f]
    else:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f]

    documents = []
    for sample in samples:
        sample_proc = preprocess_crag_sample(sample)
        for doc in sample_proc['search_results']:
            doc_proc = preprocess_crag_document(doc)
            doc_proc['timestamp'] = doc.get('page_last_modified')
            documents.append(doc_proc)
    doc_encoder = DocumentEncoder()
    texts = [doc['text'] for doc in documents]
    doc_ids = [doc['id'] for doc in documents]
    metadata = [{'timestamp': doc['timestamp'], 'text': doc['text']} for doc in documents]
    embeddings = doc_encoder.encode(texts)
    vector_store = VectorStore(embedding_dim=embeddings.shape[1])
    vector_store.add_documents(embeddings, doc_ids, metadata)
    index_path = os.path.join(output_dir, 'document_index.faiss')
    vector_store.save_index(index_path)
    print("Document index built and saved.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python build_index.py <crag_dataset.jsonl>")
        sys.exit(1)
    build_document_index(sys.argv[1])
