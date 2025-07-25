from transformers import pipeline
import torch
from typing import List

class LLMGenerator:
    def __init__(self, model_name: str = "distilbert/distilgpt2", device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.generator = pipeline("text-generation", model=model_name, device=self.device)
            print(f"LLMGenerator loaded model {model_name} on {self.device}")
        except Exception as e:
            print(f"Error loading LLM model {model_name}: {e}")
            print("Please ensure the model name is correct and you have enough resources.")
            self.generator = None

    def generate_answer(self, question: str, context: List[str], max_length: int = 200) -> str:
        if self.generator is None:
            return "Error: LLM model not loaded."
        context_str = "\n".join(context)
        prompt = f"Context: {context_str}\nQuestion: {question}\nAnswer:"
        try:
            response = self.generator(prompt, max_new_tokens=max_length, do_sample=False,
                                      num_return_sequences=1, return_full_text=False)
            if response and len(response) > 0:
                answer = response[0]['generated_text'].strip()
                if "Question:" in answer:
                    answer = answer.split("Question:")[0].strip()
                if "Answer:" in answer:
                    answer = answer.split("Answer:", 1)[-1].strip()
                return answer
            return "No answer generated."
        except Exception as e:
            return f"Error during answer generation: {e}"
