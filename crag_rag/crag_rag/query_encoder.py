from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

class NormalQueryEncoder:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"NormalQueryEncoder loaded model {model_name} on {self.device}")

    def encode(self, query_text: str) -> np.ndarray:
        embedding = self.model.encode(query_text, convert_to_numpy=True)
        return embedding

class TemporalQueryEncoder:
    def __init__(self, model_path: str = 'path/to/your/fine_tuned_contriever.bin'):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
            self.model = AutoModel.from_pretrained('facebook/contriever')
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            self.model.eval()
            print(f"TemporalQueryEncoder loaded fine-tuned Contriever from {model_path} on {self.device}")
        except Exception as e:
            print(f"Error loading fine-tuned Contriever model: {e}")
            print("Please ensure 'model_path' is correct and compatible with AutoModel.from_pretrained or provide custom loading logic.")
            self.model = None

    def encode(self, query_text: str) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TemporalQueryEncoder model not loaded.")
        inputs = self.tokenizer(query_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
        return embedding
