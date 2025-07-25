import re
from datetime import datetime

class TimeAwareModule:
    def __init__(self):
        self.temporal_keywords = [
            r'\bwhen\b', r'\bafter\b', r'\bbefore\b', r'\bduring\b', r'\bin\s+\d{4}\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b',
            r'\d{2}/\d{2}/\d{4}', r'\d{4}-\d{2}-\d{2}', r'\d{4}s\b'
        ]
        self.keyword_patterns = [re.compile(p, re.IGNORECASE) for p in self.temporal_keywords]
        print("TimeAwareModule initialized with basic temporal keyword patterns.")

    def get_temporal_score(self, text: str) -> float:
        score = 0
        for pattern in self.keyword_patterns:
            if pattern.search(text):
                score += 1
        return score / len(self.keyword_patterns)

    def is_temporal_query(self, query_text: str, threshold: float = 0.1) -> bool:
        score = self.get_temporal_score(query_text)
        return score > threshold

    def get_temporal_relevance_from_timestamps(self, query_timestamp_str: str, doc_timestamp_str: str) -> float:
        try:
            query_time = self._parse_timestamp(query_timestamp_str)
            doc_time = self._parse_timestamp(doc_timestamp_str)
            if query_time is None or doc_time is None:
                return 0.0
            time_diff = abs((query_time - doc_time).days)
            scale_factor = 365 * 5
            relevance = max(0.0, 1.0 - (time_diff / scale_factor))
            return relevance
        except Exception:
            return 0.0

    def _parse_timestamp(self, ts_str: str):
        if not ts_str:
            return None
        try:
            return datetime.strptime(ts_str, '%Y-%m-%d')
        except ValueError:
            try:
                return datetime.strptime(ts_str, '%Y-%m')
            except ValueError:
                try:
                    return datetime.strptime(ts_str, '%Y')
                except ValueError:
                    return None
