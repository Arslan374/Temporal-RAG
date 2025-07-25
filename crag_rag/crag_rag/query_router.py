from .time_aware import TimeAwareModule
from .query_encoder import NormalQueryEncoder, TemporalQueryEncoder
import numpy as np

class QueryRouter:
    def __init__(self, time_aware_module: TimeAwareModule, 
                 normal_encoder: NormalQueryEncoder, 
                 temporal_encoder: TemporalQueryEncoder,
                 temporal_threshold: float = 0.1):
        self.time_aware_module = time_aware_module
        self.normal_encoder = normal_encoder
        self.temporal_encoder = temporal_encoder
        self.temporal_threshold = temporal_threshold
        print("QueryRouter initialized.")

    def route_and_encode(self, query_text: str) -> np.ndarray:
        if self.time_aware_module.is_temporal_query(query_text, self.temporal_threshold):
            print(f"Query '{query_text}' identified as temporal. Using TemporalQueryEncoder.")
            return self.temporal_encoder.encode(query_text)
        else:
            print(f"Query '{query_text}' identified as non-temporal. Using NormalQueryEncoder.")
            return self.normal_encoder.encode(query_text)
