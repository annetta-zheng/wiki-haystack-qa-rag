#!/usr/bin/env python3
import json
import os
import time
from typing import Dict, List, Optional
import concurrent.futures

import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Evaluator:
    def __init__(self, qa_set: List[Dict], verbose: bool = True, api_key: Optional[str] = None):
        """Initialize the evaluator with a Q&A set."""
        self.qa_set = qa_set
        self.verbose = verbose
        
        # Set up OpenAI client
        api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OpenAI API key must be provided via parameter or OPENAI_API_KEY environment variable. "
                "Please set it in your .env file or environment."
            )
        self.client = openai.OpenAI(api_key=api_key)
        
        # ANSI color codes for output formatting
        self.colors = {
            'cyan': '\033[96m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'purple': '\033[95m',
            'blue': '\033[94m',
            'bold': '\033[1m',
            'end': '\033[0m'
        }
    
    def run_answer_parallel(self, answer_function, file_path: str, max_workers: int = 5):
        """Run answer function for all questions in parallel."""
        results = {}
        
        def get_answer(qa_item: Dict) -> Dict:
            """Worker function to get answer for a single question."""
            question = qa_item["question"]
            try:
                start = time.time()
                answer = answer_function(question, file_path)
                elapsed = time.time() - start
                return {
                    "question": question,
                    "expected": qa_item["answer"],
                    "actual": answer,
                    "difficulty": qa_item.get("difficulty", "unknown"),
                    "time": elapsed,
                    "error": None
                }
            except Exception as e:
                return {
                    "question": question,
                    "expected": qa_item["answer"],
                    "actual": None,
                    "difficulty": qa_item.get("difficulty", "unknown"),
                    "time": 0,
                    "error": str(e)
                }
        
        if self.verbose:
            print(f"\n{self.colors['cyan']}{self.colors['bold']}PARALLEL EVALUATION{self.colors['end']}")
            print(f"{self.colors['blue']}Running {len(self.qa_set)} questions with {max_workers} workers{self.colors['end']}")
            print(f"{'─' * 50}")
        
        # Run all questions in parallel
        max_workers = min(max_workers or 5, 10)  # Cap at 10 workers to avoid overwhelming the system
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_qa = {executor.submit(get_answer, qa): qa for qa in self.qa_set}
            
            # Process results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_qa):
                result = future.result()
                results[result["question"]] = result
                completed += 1
                
                if self.verbose:
                    status = "✓" if result["error"] is None else "✗"
                    color = self.colors['green'] if result["error"] is None else self.colors['red']
                    
                    # Enhanced logging with full details
                    print(f"\n{color}Question #{completed}:{self.colors['end']}")
                    print(f"{color}Q: {result['question']}{self.colors['end']}")
                    print(f"{color}Duration: {result['time']:.2f}s{self.colors['end']}")
                    print(f"{color}Expected: {result['expected']}{self.colors['end']}")
                    
                    if result["error"] is None:
                        print(f"{color}Answer: {result['actual']}{self.colors['end']}")
                        print(f"{color}Status: {status} COMPLETED{self.colors['end']}")
                    else:
                        print(f"{self.colors['red']}Error: {result['error']}{self.colors['end']}")
                        print(f"{self.colors['red']}Status: {status} FAILED{self.colors['end']}")
                    
                    print(f"{'-' * 60}")
        
        return results
    
    def grade_with_model(self, results: Dict) -> Dict:
        """Use OpenAI to grade all answers in a single call."""
        if self.verbose:
            print(f"\n{self.colors['cyan']}{self.colors['bold']}MODEL GRADING{self.colors['end']}")
            print(f"{self.colors['blue']}Sending {len(results)} results to model for grading...{self.colors['end']}")

        grading_data = []
        for question, result in results.items():
            actual_answer = result["actual"] if result["error"] is None else f"ERROR: {result['error']}"
            grading_data.append({
                "question": question,
                "expected_answer": result["expected"],
                "actual_answer": actual_answer,
                "difficulty": result["difficulty"]
            })

        prompt = f"""Grade each answer below. The correctness rule depends on what expected_answer is.

CASE 1 - expected_answer is a specific fact (a name, date, place, number, etc.):
Mark CORRECT if actual_answer contains the same core fact, allowing minor variations:
- Missing titles/honorifics: "Magnus Eriksson" matches "Captain Magnus Eriksson"
- Missing prefixes: "Lincoln" matches "Abraham Lincoln"
- Partial dates: "267 CE" matches "267-270 CE"
- Unit variations: "2,000 kilometers" matches "2,000-kilometer" or "2000 km"
- Minor formatting: "Dr. Smith" matches "Dr Smith"
- Partial names in context: "Gettysburg" matches "Battle of Gettysburg"
- Numeric words vs digits: "first" matches "1st", "twenty" matches "20"

Mark INCORRECT in CASE 1 if:
- The core fact is wrong (different name, wrong number, wrong date)
- The core fact is missing entirely
- actual_answer is "I don't know" / "unknown" / "not stated" / any phrasing indicating no answer
  (the candidate must state the actual fact - refusing to answer is NOT correct)

CASE 2 - expected_answer is exactly the string "I don't know":
Mark CORRECT only if actual_answer clearly conveys that the information is unavailable.
Accept: "I don't know", "unknown", "unclear", "not stated", "not available",
"not specified", "not mentioned", "no information", "insufficient information",
"cannot be determined", "N/A".
Mark INCORRECT if actual_answer gives any specific fact - a number, name, date, or
description - even if it sounds plausible. The candidate is being tested on whether
they correctly recognized the information was not in the source.

Examples of CORRECT case-1 pairs:
- expected="Skyfather's Breath", actual="Skyfather's Breath" -> correct
- expected="Skyfather's Breath", actual="The Skyfather's Breath" -> correct

Examples of INCORRECT case-1 pairs:
- expected="Skyfather's Breath", actual="I don't know" -> INCORRECT (refusal is not a match)
- expected="Skyfather's Breath", actual="Windbreath" -> INCORRECT (wrong fact)
- expected="3000", actual="I don't know" -> INCORRECT

Examples of CORRECT case-2 pairs:
- expected="I don't know", actual="I don't know" -> correct
- expected="I don't know", actual="Not stated in the article" -> correct

Examples of INCORRECT case-2 pairs:
- expected="I don't know", actual="500 meters" -> INCORRECT (fabricated a specific value)
- expected="I don't know", actual="approximately 2000" -> INCORRECT

Return JSON: {{"results": [{{"question": "...", "is_correct": bool, "reason": "brief explanation"}}]}}

Questions:
{json.dumps(grading_data, separators=(',', ':'))}"""

        response = self.client.chat.completions.create(
            model="gpt-4.1",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a strict but fair grading assistant. Grade each answer using the two-case rule the user provides. Do not treat refusals ('I don't know') as correct unless the expected answer is also 'I don't know'. Return a JSON object with a 'results' key containing an array of grading results."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
        )

        raw_response = response.choices[0].message.content
        if not raw_response:
            raise ValueError("Empty response from model")
        grading_results = json.loads(raw_response)
        grading_results = self._extract_results_list(grading_results)
        grading_lookup = {item["question"]: item for item in grading_results}

        for question, result in results.items():
            if question in grading_lookup:
                result["is_correct"] = grading_lookup[question]["is_correct"]
                result["grading_reason"] = grading_lookup[question].get("reason", "")
            else:
                result["is_correct"] = False
                result["grading_reason"] = "Not graded"

        if self.verbose:
            print(f"{self.colors['green']}✓ Model grading complete{self.colors['end']}")
        
    
        return results
    
    def _extract_results_list(self, grading_results):
        """Extract results list from model response."""
        if isinstance(grading_results, dict) and "results" in grading_results:
            grading_results = grading_results["results"]
        elif isinstance(grading_results, dict):
            # Look for any key that contains an array
            for value in grading_results.values():
                if isinstance(value, list):
                    grading_results = value
                    break
            else:
                # Try to extract values as a list
                if grading_results:
                    grading_results = list(grading_results.values()) if all(isinstance(v, dict) for v in grading_results.values()) else [grading_results]
        
        if not isinstance(grading_results, list):
            raise ValueError(f"Expected list from model, got {type(grading_results)}")
        
        return grading_results
    
    def _fallback_grading(self, results: Dict) -> None:
        """Apply simple exact match grading as fallback."""
        for result in results.values():
            if result["error"] is None and result["actual"]:
                normalized_expected = result["expected"].lower().strip().rstrip('.!?')
                normalized_actual = result["actual"].lower().strip().rstrip('.!?')
                result["is_correct"] = normalized_expected == normalized_actual
                result["grading_reason"] = "Exact match" if result["is_correct"] else "No match"
            else:
                result["is_correct"] = False
                result["grading_reason"] = "Error or no answer"
    
    def run_evaluation(self, answer_function, file_path: str, max_workers: Optional[int] = None) -> Dict:
        """Run the complete evaluation with parallel execution and model grading."""
        start_time = time.time()
        
        # Step 1: Run all questions in parallel
        results = self.run_answer_parallel(answer_function, file_path, max_workers)
        
        # Step 2: Grade all answers with a single model call
        results = self.grade_with_model(results)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Calculate statistics
        total_questions = len(results)
        correct_answers = sum(1 for r in results.values() if r.get("is_correct", False))
        accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        
        # Group results by difficulty
        results_by_difficulty = {}
        for result in results.values():
            difficulty = result.get("difficulty", "unknown")
            if difficulty not in results_by_difficulty:
                results_by_difficulty[difficulty] = {"correct": 0, "total": 0}
            results_by_difficulty[difficulty]["total"] += 1
            if result.get("is_correct", False):
                results_by_difficulty[difficulty]["correct"] += 1
        
        # Convert results dict to list for detailed_results
        detailed_results = list(results.values())
        
        summary = {
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "accuracy": accuracy,
            "execution_time": execution_time,
            "results_by_difficulty": results_by_difficulty,
            "detailed_results": detailed_results
        }
        
        self._print_results(summary, detailed_results)
        return summary
    
    def _print_results(self, summary: Dict, detailed_results: List[Dict]) -> None:
        """Print evaluation results."""
        if not self.verbose:
            return
            
        print(f"\n{'─' * 50}")
        print(f"{self.colors['cyan']}{self.colors['bold']}RESULTS{self.colors['end']}")
        
        # Show incorrect answers
        incorrect = [r for r in detailed_results if not r.get("is_correct", False)]
        if incorrect:
            print(f"\n{self.colors['yellow']}{self.colors['bold']}Incorrect Answers:{self.colors['end']}")
            for r in incorrect[:10]:  # Show first 10 incorrect
                expected_key = "expected_answer" if "expected_answer" in r else "expected"
                actual_key = "actual_answer" if "actual_answer" in r else "actual"
                print(f"  Q: {r['question'][:60]}...")
                print(f"     Expected: {self.colors['green']}{r[expected_key]}{self.colors['end']}")
                print(f"     Got:      {self.colors['red']}{r[actual_key]}{self.colors['end']}")
                print(f"     Reason:   {self.colors['yellow']}{r.get('reason', r.get('grading_reason', 'N/A'))}{self.colors['end']}")
            if len(incorrect) > 10:
                print(f"  ... and {len(incorrect) - 10} more")
        
        accuracy = summary["accuracy"]

        print(f"\n{self.colors['bold']}Overall Performance:{self.colors['end']}")
        print(f"Total Questions: {self.colors['blue']}{summary['total_questions']}{self.colors['end']}")
        print(f"Correct Answers: {self.colors['green']}{summary['correct_answers']}{self.colors['end']}")
        print(f"Accuracy: {self.colors['purple']}{accuracy:.1f}%{self.colors['end']}")
        print(f"Execution Time: {self.colors['cyan']}{summary['execution_time']:.2f}s{self.colors['end']}")
        
        print(f"\n{self.colors['bold']}Results by Difficulty:{self.colors['end']}")
        difficulty_order = ["easy", "medium", "hard", "unknown"]
        for difficulty in difficulty_order:
            if difficulty in summary["results_by_difficulty"]:
                stats = summary["results_by_difficulty"][difficulty]
                difficulty_accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
                if difficulty_accuracy >= 80:
                    color = self.colors['green']
                elif difficulty_accuracy >= 60:
                    color = self.colors['yellow']
                else:
                    color = self.colors['red']
                print(f"  {difficulty:8s}: {color}{stats['correct']:2d}/{stats['total']:2d} ({difficulty_accuracy:5.1f}%){self.colors['end']}")
        
        print(f"{'─' * 50}")

def load_example_qa(file_path: str) -> List[Dict]:
    """Load the example Q&A set."""
    with open(file_path, 'r') as f:
        return json.load(f)

