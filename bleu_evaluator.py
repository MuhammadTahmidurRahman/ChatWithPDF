import json
import os
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from typing import List, Dict, Any, Optional
import threading

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class BLEUEvaluator:
    """
    BLEU Score evaluator for PDF chat responses
    """
    
    def __init__(self, test_data_path: str = "test_dataset.json"):
        self.test_data_path = test_data_path
        self.reference_data = self._load_reference_data()
        self.smoothing = SmoothingFunction()
        self._lock = threading.Lock()
        
    def _load_reference_data(self) -> Dict[str, List[str]]:
        """Load reference data from JSON file"""
        try:
            if not os.path.exists(self.test_data_path):
                print(f"Warning: Test data file not found at {self.test_data_path}")
                return {}
                
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Convert to question -> references mapping
            reference_dict = {}
            for item in data:
                question = self._preprocess_text(item.get('question', ''))  # Preprocess the question key
                references = item.get('references', [])
                if question and references:
                    reference_dict[question] = references
                    
            print(f"✅ Loaded {len(reference_dict)} reference questions for BLEU evaluation")
            return reference_dict
            
        except Exception as e:
            print(f"Error loading reference data: {str(e)}")
            return {}
    
    def _preprocess_text(self, text: str) -> str:
        """Less aggressive preprocessing for better matching"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation but keep important words
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for BLEU calculation"""
        processed_text = self._preprocess_text(text)
        try:
            tokens = word_tokenize(processed_text)
            return [token for token in tokens if token.strip()]
        except:
            # Fallback to simple split if NLTK tokenizer fails
            return processed_text.split()
    
    def _find_best_reference_match(self, question: str) -> List[str]:
        """Find the best matching reference answers for a question"""
        question_clean = self._preprocess_text(question)
        
        # Exact match first
        if question_clean in self.reference_data:
            return self.reference_data[question_clean]
        
        # Enhanced partial matching with better similarity scoring
        best_matches = []
        best_score = 0
        question_words = set(question_clean.split())
        
        for ref_question, ref_answers in self.reference_data.items():
            ref_words = set(ref_question.split())
            
            # Calculate Jaccard similarity (intersection over union)
            intersection = question_words & ref_words
            union = question_words | ref_words
            
            if len(union) > 0:
                jaccard_similarity = len(intersection) / len(union)
                
                # Also check for key phrase matching
                key_phrases_match = any(
                    phrase in question_clean for phrase in ref_question.split() 
                    if len(phrase) > 3
                )
                
                # Combined similarity score
                total_similarity = jaccard_similarity + (0.2 if key_phrases_match else 0)
                
                if total_similarity > best_score and total_similarity > 0.25:  # Lower threshold
                    best_score = total_similarity
                    best_matches = ref_answers
        
        # Debug logging
        if best_matches:
            print(f"BLEU: Found match with similarity {best_score:.3f} for question: {question[:50]}...")
        else:
            print(f"BLEU: No reference match found for question: {question[:50]}...")
        
        return best_matches
    
    def debug_question_matching(self, question: str):
        """Debug function to see why a question isn't matching"""
        print(f"\n=== DEBUG: Question Matching ===")
        print(f"Original question: '{question}'")
        
        processed = self._preprocess_text(question)
        print(f"Processed question: '{processed}'")
        
        # Check exact match
        if processed in self.reference_data:
            print(f"Exact match found: {self.reference_data[processed]}")
            return
        
        # Check partial matches
        print("Checking partial matches...")
        question_words = set(processed.split())
        
        best_match = None
        best_score = 0
        
        for ref_question, ref_answers in self.reference_data.items():
            ref_words = set(ref_question.split())
            
            intersection = question_words & ref_words
            union = question_words | ref_words
            
            if len(union) > 0:
                jaccard_similarity = len(intersection) / len(union)
                
                if jaccard_similarity > best_score:
                    best_score = jaccard_similarity
                    best_match = (ref_question, ref_answers)
        
        if best_match and best_score > 0:
            print(f"Best match: '{best_match[0]}' with similarity {best_score:.3f}")
            print(f"References: {best_match[1]}")
        else:
            print("No match found")
        
        print("=== END DEBUG ===\n")
    
    def calculate_bleu_score(self, question: str, generated_answer: str) -> Dict[str, Any]:
        """
        Calculate BLEU score for a generated answer against reference answers
        """
        with self._lock:
            try:
                # Find reference answers for this question
                reference_answers = self._find_best_reference_match(question)
                
                if not reference_answers:
                    # Debug why no match was found
                    self.debug_question_matching(question)
                    return {
                        'bleu_score': 0.0,
                        'bleu_1': 0.0,
                        'bleu_2': 0.0,
                        'bleu_3': 0.0,
                        'bleu_4': 0.0,
                        'reference_found': False,
                        'num_references': 0,
                        'explanation': 'No reference answer found for this question'
                    }
                
                print(f"BLEU: Found {len(reference_answers)} reference answers")
                
                # Tokenize generated answer
                generated_tokens = self._tokenize_text(generated_answer)
                
                if not generated_tokens:
                    return {
                        'bleu_score': 0.0,
                        'bleu_1': 0.0,
                        'bleu_2': 0.0,
                        'bleu_3': 0.0,
                        'bleu_4': 0.0,
                        'reference_found': True,
                        'num_references': len(reference_answers),
                        'explanation': 'Generated answer is empty after tokenization'
                    }
                
                # Tokenize all reference answers
                reference_tokens_list = []
                for ref_answer in reference_answers:
                    ref_tokens = self._tokenize_text(ref_answer)
                    if ref_tokens:
                        reference_tokens_list.append(ref_tokens)
                
                if not reference_tokens_list:
                    return {
                        'bleu_score': 0.0,
                        'bleu_1': 0.0,
                        'bleu_2': 0.0,
                        'bleu_3': 0.0,
                        'bleu_4': 0.0,
                        'reference_found': True,
                        'num_references': len(reference_answers),
                        'explanation': 'Reference answers are empty after processing'
                    }
                
                # Calculate BLEU scores with smoothing
                try:
                    # Individual n-gram BLEU scores
                    bleu_1 = sentence_bleu(reference_tokens_list, generated_tokens, 
                                         weights=(1, 0, 0, 0), 
                                         smoothing_function=self.smoothing.method1)
                    
                    bleu_2 = sentence_bleu(reference_tokens_list, generated_tokens, 
                                         weights=(0.5, 0.5, 0, 0), 
                                         smoothing_function=self.smoothing.method1)
                    
                    bleu_3 = sentence_bleu(reference_tokens_list, generated_tokens, 
                                         weights=(0.33, 0.33, 0.33, 0), 
                                         smoothing_function=self.smoothing.method1)
                    
                    bleu_4 = sentence_bleu(reference_tokens_list, generated_tokens, 
                                         weights=(0.25, 0.25, 0.25, 0.25), 
                                         smoothing_function=self.smoothing.method1)
                    
                    # Overall BLEU score (BLEU-4 is standard)
                    overall_bleu = bleu_4
                    
                    return {
                        'bleu_score': round(overall_bleu, 4),
                        'bleu_1': round(bleu_1, 4),
                        'bleu_2': round(bleu_2, 4),
                        'bleu_3': round(bleu_3, 4),
                        'bleu_4': round(bleu_4, 4),
                        'reference_found': True,
                        'num_references': len(reference_answers),
                        'explanation': f'BLEU calculated against {len(reference_answers)} reference(s)'
                    }
                    
                except Exception as e:
                    print(f"Error calculating BLEU scores: {str(e)}")
                    return {
                        'bleu_score': 0.0,
                        'bleu_1': 0.0,
                        'bleu_2': 0.0,
                        'bleu_3': 0.0,
                        'bleu_4': 0.0,
                        'reference_found': True,
                        'num_references': len(reference_answers),
                        'explanation': f'Error in BLEU calculation: {str(e)}'
                    }
                    
            except Exception as e:
                print(f"Error in BLEU evaluation: {str(e)}")
                return {
                    'bleu_score': 0.0,
                    'bleu_1': 0.0,
                    'bleu_2': 0.0,
                    'bleu_3': 0.0,
                    'bleu_4': 0.0,
                    'reference_found': False,
                    'num_references': 0,
                    'explanation': f'BLEU evaluation error: {str(e)}'
                }
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation capabilities"""
        return {
            'total_reference_questions': len(self.reference_data),
            'test_data_path': self.test_data_path,
            'test_data_exists': os.path.exists(self.test_data_path),
            'available_questions': list(self.reference_data.keys())[:5]  # First 5 for preview
        }
    
    def batch_evaluate(self, qa_pairs: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Evaluate multiple question-answer pairs and return aggregate statistics
        """
        results = []
        total_bleu = 0.0
        valid_evaluations = 0
        
        for qa_pair in qa_pairs:
            question = qa_pair.get('question', '')
            answer = qa_pair.get('answer', '')
            
            if question and answer:
                bleu_result = self.calculate_bleu_score(question, answer)
                results.append({
                    'question': question,
                    'bleu_score': bleu_result['bleu_score'],
                    'reference_found': bleu_result['reference_found']
                })
                
                if bleu_result['reference_found'] and bleu_result['bleu_score'] > 0:
                    total_bleu += bleu_result['bleu_score']
                    valid_evaluations += 1
        
        avg_bleu = total_bleu / valid_evaluations if valid_evaluations > 0 else 0.0
        
        return {
            'total_questions': len(qa_pairs),
            'valid_evaluations': valid_evaluations,
            'average_bleu': round(avg_bleu, 4),
            'detailed_results': results
        }

# Global instance for use across the application
bleu_evaluator = None

def get_bleu_evaluator() -> BLEUEvaluator:
    """Get or create global BLEU evaluator instance"""
    global bleu_evaluator
    if bleu_evaluator is None:
        bleu_evaluator = BLEUEvaluator()
    return bleu_evaluator

def evaluate_answer_bleu(question: str, answer: str) -> Dict[str, Any]:
    """
    Convenience function to evaluate a single answer
    """
    evaluator = get_bleu_evaluator()
    return evaluator.calculate_bleu_score(question, answer)

# Test function
def test_bleu_evaluator():
    """Test the BLEU evaluator with sample data"""
    evaluator = BLEUEvaluator()
    
    # Test question from the dataset
    test_question = "What is the main theme of The Alchemist?"
    test_answer = "The main theme is about pursuing one's personal legend and following dreams with the universe's help."
    
    result = evaluator.calculate_bleu_score(test_question, test_answer)
    
    print("BLEU Evaluator Test Results:")
    print(f"Question: {test_question}")
    print(f"Answer: {test_answer}")
    print(f"BLEU Score: {result['bleu_score']}")
    print(f"Reference Found: {result['reference_found']}")
    print(f"Explanation: {result['explanation']}")
    
    return result

if __name__ == "__main__":
    # Run test when script is executed directly
    test_bleu_evaluator()