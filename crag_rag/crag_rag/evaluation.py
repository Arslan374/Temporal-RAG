from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Dict, Union
from .simple_rag import SimpleRAG
from .temporal_rag import TemporalRAGPipeline
from .time_aware import TimeAwareModule

class Evaluator:
    def __init__(self):
        print("Evaluator initialized.")

    def calculate_metrics(self, predictions: List[str], ground_truths: List[str], questions: List[Dict]) -> Dict:
        results = {}
        correct_answers = 0
        for pred, gt in zip(predictions, ground_truths):
            if pred.strip().lower() == gt.strip().lower():
                correct_answers += 1
        results['exact_match_accuracy'] = correct_answers / len(predictions) if predictions else 0
        print("\n--- Evaluation Results ---")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        return results

    def evaluate_temporal_classification(self, time_aware_module: TimeAwareModule, temporal_test_questions: List[Dict]) -> Dict:
        if not temporal_test_questions:
            print("No temporal test questions provided for TimeAwareModule evaluation.")
            return {}
        true_labels = [q['ground_truth_temporal'] for q in temporal_test_questions if q['ground_truth_temporal'] is not None]
        if not true_labels:
            print("No ground truth temporal labels found in the test questions.")
            return {}
        predictions = [time_aware_module.is_temporal_query(q['text']) for q in temporal_test_questions if q['ground_truth_temporal'] is not None]
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, zero_division=0),
            'recall': recall_score(true_labels, predictions, zero_division=0),
            'f1_score': f1_score(true_labels, predictions, zero_division=0)
        }
        print("\n--- TimeAwareModule Temporal Classification Results ---")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        return metrics

    def run_evaluation_suite(self, 
                             model_or_pipeline_instance: Union[SimpleRAG, TemporalRAGPipeline], 
                             test_dataset: List[Dict]) -> Dict:
        print(f"\n--- Running Evaluation for: {model_or_pipeline_instance.__class__.__name__} ---")
        predictions = []
        ground_truths = []
        for i, q_data in enumerate(test_dataset):
            question_text = q_data['text']
            ground_truth_answer = q_data['answer']
            print(f"Processing question {i+1}/{len(test_dataset)}: {question_text[:50]}...")
            predicted_answer = model_or_pipeline_instance.answer_question(question_text)
            predictions.append(predicted_answer)
            ground_truths.append(ground_truth_answer)
        results = self.calculate_metrics(predictions, ground_truths, test_dataset)
        return results
