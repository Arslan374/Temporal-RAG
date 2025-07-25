import json
from typing import List, Dict
import os

def load_json_dataset(filepath: str) -> List[Dict]:
    """Loads a dataset from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def preprocess_crag_document(doc: Dict) -> Dict:
    """
    Preprocess a CRAG document (search result page).
    """
    doc_id = doc.get('page_url') or doc.get('page_name')
    text = doc.get('page_snippet', '').strip()
    timestamp = doc.get('page_last_modified')
    return {'id': doc_id, 'text': text, 'timestamp': timestamp}

def preprocess_crag_sample(sample: Dict) -> Dict:
    """
    Preprocess a CRAG dataset sample (question + search results).
    """
    return {
        'id': sample.get('interaction_id'),
        'text': sample.get('query', '').strip(),
        'answer': sample.get('answer', ''),
        'alt_ans': sample.get('alt_ans', []),
        'timestamp': sample.get('query_time'),
        'search_results': sample.get('search_results', []),
        'domain': sample.get('domain'),
        'question_type': sample.get('question_type'),
        'static_or_dynamic': sample.get('static_or_dynamic'),
        'split': sample.get('split'),
    }
