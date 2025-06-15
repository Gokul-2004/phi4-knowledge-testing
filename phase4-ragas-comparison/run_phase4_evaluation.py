#!/usr/bin/env python3
"""
Phase 4: Enhanced RAGAS-Based GPT-4o Mini vs Phi4 Comparison
With comprehensive metrics, full response storage, and side-by-side analysis
"""

import json
import time
import subprocess
import openai
import os
import hashlib
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from datetime import datetime
import re

# Initialize LangWatch with error handling
LANGWATCH_AVAILABLE = False
try:
    import langwatch
    
    if hasattr(langwatch, 'configure') and os.getenv('LANGWATCH_API_KEY'):
        langwatch.configure(api_key=os.getenv('LANGWATCH_API_KEY'))
        LANGWATCH_AVAILABLE = True
        print("✅ LangWatch configured successfully")
    elif os.getenv('LANGWATCH_API_KEY'):
        os.environ['LANGWATCH_API_KEY'] = os.getenv('LANGWATCH_API_KEY')
        LANGWATCH_AVAILABLE = True
        print("✅ LangWatch API key configured")
    else:
        print("⚠️  LangWatch API key not found - tracking disabled")
        
except ImportError:
    print("⚠️  LangWatch not available - install with: pip install langwatch")
except Exception as e:
    print(f"⚠️  LangWatch setup failed: {e}")


class ResponseAnalyzer:
    """Analyze response content for patterns and quality indicators"""
    
    @staticmethod
    def analyze_uncertainty(text: str) -> Dict:
        """Analyze uncertainty expressions in response"""
        uncertainty_phrases = [
            "don't know", "i don't know", "not sure", "uncertain",
            "don't have", "cannot provide", "can't say", "unclear",
            "recommend checking", "consult", "refer to", "check the",
            "as far as i know", "to my knowledge", "i believe",
            "might", "could", "possibly", "perhaps", "may be"
        ]
        
        confidence_phrases = [
            "definitely", "certainly", "absolutely", "clearly",
            "obviously", "without doubt", "precisely", "exactly",
            "specifically", "explicitly", "undoubtedly"
        ]
        
        text_lower = text.lower()
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in text_lower)
        confidence_count = sum(1 for phrase in confidence_phrases if phrase in text_lower)
        
        says_dont_know = any(phrase in text_lower for phrase in ["don't know", "i don't know", "don't have specific"])
        
        return {
            "uncertainty_phrases_found": [phrase for phrase in uncertainty_phrases if phrase in text_lower],
            "confidence_phrases_found": [phrase for phrase in confidence_phrases if phrase in text_lower],
            "uncertainty_score": uncertainty_count / (uncertainty_count + confidence_count + 1),
            "says_dont_know": says_dont_know,
            "confidence_level": "appropriately_uncertain" if says_dont_know else ("overconfident" if confidence_count > uncertainty_count else "moderate")
        }
    
    @staticmethod
    def detect_fabrication_indicators(text: str) -> List[str]:
        """Detect potential fabrication indicators"""
        indicators = []
        
        # Specific technical details that might be fabricated
        if re.search(r'(field|parameter|option) (can be set to|accepts|supports) ["\'][^"\']+["\']', text):
            indicators.append("specific_field_values")
            
        if re.search(r'```[\s\S]*```', text):
            indicators.append("code_example")
            
        if re.search(r'(introduced in|added in|available since) (version |v)?[\d\.]+', text):
            indicators.append("version_specific_claims")
            
        if re.search(r'(example|for instance|such as):?\s*\n', text):
            indicators.append("detailed_examples")
            
        # Overly specific numeric values
        if re.search(r'\b\d+\.\d+\.\d+\b', text) and "kubernetes" in text.lower():
            indicators.append("specific_version_numbers")
            
        return indicators
    
    @staticmethod
    def count_citations(text: str) -> Dict:
        """Count and analyze citations/references"""
        citation_patterns = [
            "official documentation", "release notes", "kubernetes docs",
            "check the", "refer to", "consult", "see the", "visit"
        ]
        
        citations_found = [pattern for pattern in citation_patterns if pattern in text.lower()]
        
        return {
            "provides_citations": len(citations_found) > 0,
            "citation_phrases": citations_found,
            "citation_count": len(citations_found)
        }


class Phi4Model:
    """Enhanced Phi4 model interface with comprehensive caching"""
    
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.cache = self.load_cache()
        
    def load_cache(self) -> Dict:
        if Path(self.cache_file).exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_cache(self):
        Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def _get_cache_key(self, question: str) -> str:
        """Generate consistent cache key"""
        return f"phi4_{hashlib.md5(question.encode()).hexdigest()[:12]}"
    
    def query(self, question: str) -> Dict:
        """Query Phi4 with comprehensive response analysis"""
        cache_key = self._get_cache_key(question)
        
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            print("💾 Using cached Phi4 response")
            return cached
        
        print("🤖 Querying Phi4...")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                ['ollama', 'run', 'phi4:latest', question],
                capture_output=True, text=True, timeout=300
            )
            response_time = round(time.time() - start_time, 2)
            
            if result.returncode == 0:
                response_text = result.stdout.strip()
                
                # Comprehensive analysis
                uncertainty_analysis = ResponseAnalyzer.analyze_uncertainty(response_text)
                fabrication_indicators = ResponseAnalyzer.detect_fabrication_indicators(response_text)
                citation_analysis = ResponseAnalyzer.count_citations(response_text)
                
                response_data = {
                    "full_text": response_text,
                    "response_time": response_time,
                    "word_count": len(response_text.split()),
                    "character_count": len(response_text),
                    "says_dont_know": uncertainty_analysis["says_dont_know"],
                    "confidence_level": uncertainty_analysis["confidence_level"],
                    "uncertainty_score": uncertainty_analysis["uncertainty_score"],
                    "uncertainty_phrases": uncertainty_analysis["uncertainty_phrases_found"],
                    "fabrication_indicators": fabrication_indicators,
                    "citations_provided": citation_analysis["provides_citations"],
                    "citation_phrases": citation_analysis["citation_phrases"],
                    "timestamp": datetime.now().isoformat(),
                    "model": "phi4:latest"
                }
                
                # Cache result
                self.cache[cache_key] = response_data
                self.save_cache()
                
                return response_data
            else:
                error_response = {
                    "full_text": f"ERROR: {result.stderr}",
                    "response_time": response_time,
                    "error": True,
                    "timestamp": datetime.now().isoformat()
                }
                return error_response
                
        except subprocess.TimeoutExpired:
            response_time = round(time.time() - start_time, 2)
            return {
                "full_text": "ERROR: Query timeout after 5 minutes",
                "response_time": response_time,
                "error": True,
                "timeout": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            response_time = round(time.time() - start_time, 2)
            return {
                "full_text": f"ERROR: {e}",
                "response_time": response_time,
                "error": True,
                "exception": str(e),
                "timestamp": datetime.now().isoformat()
            }


class GPT4oMiniModel:
    """Enhanced GPT-4o Mini model interface with LangWatch integration"""
    
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.cache = self.load_cache()
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def load_cache(self) -> Dict:
        if Path(self.cache_file).exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_cache(self):
        Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def _get_cache_key(self, question: str) -> str:
        """Generate consistent cache key"""
        return f"gpt4o_{hashlib.md5(question.encode()).hexdigest()[:12]}"
    
    def query(self, question: str) -> Dict:
        """Query GPT-4o Mini with LangWatch tracking and comprehensive analysis"""
        cache_key = self._get_cache_key(question)
        
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            print("💾 Using cached GPT-4o Mini response")
            return cached
        
        print("🤖 Querying GPT-4o Mini...")
        start_time = time.time()
        
        # Enhanced system prompt for consistency
        system_prompt = (
            "You are a Kubernetes expert. Provide detailed, accurate responses. "
            "IMPORTANT: If you don't have specific information about Kubernetes v1.32.0 features, "
            "explicitly say 'I don't know' rather than guessing or fabricating information. "
            "Be honest about knowledge limitations."
        )
        
        def make_openai_call():
            return self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_tokens=1500,
                temperature=0.1
            )
        
        try:
            # Use LangWatch tracing if available
            if LANGWATCH_AVAILABLE and hasattr(langwatch, 'trace'):
                @langwatch.trace()
                def traced_openai_call():
                    return make_openai_call()
                response = traced_openai_call()
            else:
                response = make_openai_call()
                
            response_time = round(time.time() - start_time, 2)
            response_text = response.choices[0].message.content.strip()
            
            # Calculate costs (approximate)
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            # GPT-4o-mini pricing: $0.00015 per 1K input tokens, $0.0006 per 1K output tokens
            cost_usd = (prompt_tokens * 0.00015 / 1000) + (completion_tokens * 0.0006 / 1000)
            
            # Comprehensive analysis
            uncertainty_analysis = ResponseAnalyzer.analyze_uncertainty(response_text)
            fabrication_indicators = ResponseAnalyzer.detect_fabrication_indicators(response_text)
            citation_analysis = ResponseAnalyzer.count_citations(response_text)
            
            response_data = {
                "full_text": response_text,
                "response_time": response_time,
                "word_count": len(response_text.split()),
                "character_count": len(response_text),
                "tokens_used": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": round(cost_usd, 6),
                "says_dont_know": uncertainty_analysis["says_dont_know"],
                "confidence_level": uncertainty_analysis["confidence_level"],
                "uncertainty_score": uncertainty_analysis["uncertainty_score"],
                "uncertainty_phrases": uncertainty_analysis["uncertainty_phrases_found"],
                "fabrication_indicators": fabrication_indicators,
                "citations_provided": citation_analysis["provides_citations"],
                "citation_phrases": citation_analysis["citation_phrases"],
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-4o-mini"
            }
            
            # Add LangWatch metadata if available
            if LANGWATCH_AVAILABLE:
                try:
                    if hasattr(langwatch, 'get_current_trace'):
                        langwatch.get_current_trace().update(
                            metadata={
                                "evaluation_type": "kubernetes_v1.32.0_expert",
                                "phase": "phase4_ragas_comparison",
                                "response_time": response_time,
                                "cost_usd": cost_usd,
                                "fabrication_risk": len(fabrication_indicators) > 0,
                                "says_dont_know": uncertainty_analysis["says_dont_know"]
                            }
                        )
                except:
                    pass
            
            # Cache result
            self.cache[cache_key] = response_data
            self.save_cache()
            
            return response_data
            
        except Exception as e:
            response_time = round(time.time() - start_time, 2)
            return {
                "full_text": f"ERROR: {e}",
                "response_time": response_time,
                "error": True,
                "exception": str(e),
                "timestamp": datetime.now().isoformat()
            }


class RAGASEvaluator:
    """Enhanced RAGAS evaluation with comprehensive metrics"""
    
    def __init__(self):
        self.ragas_available = self._check_ragas()
        
    def _check_ragas(self) -> bool:
        """Check if RAGAS is available"""
        try:
            import ragas
            from ragas.metrics import (
                faithfulness, answer_relevancy, answer_correctness,
                answer_semantic_similarity, context_precision, context_recall
            )
            return True
        except ImportError:
            print("⚠️  RAGAS not available - install with: pip install ragas datasets")
            return False
        except Exception as e:
            print(f"⚠️  RAGAS import failed: {e}")
            return False
    
    def evaluate_responses(self, phi4_results: List[Dict], gpt4o_results: List[Dict]) -> Dict:
        """Evaluate responses using comprehensive RAGAS metrics"""
        if not self.ragas_available:
            return {"error": "RAGAS not available"}
        
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness, answer_relevancy, answer_correctness,
                answer_semantic_similarity
            )
            from datasets import Dataset
            
            print("📊 Preparing datasets for comprehensive RAGAS evaluation...")
            
            # Filter valid responses
            phi4_valid = [r for r in phi4_results if not r.get('error', False)]
            gpt4o_valid = [r for r in gpt4o_results if not r.get('error', False)]
            
            if not phi4_valid or not gpt4o_valid:
                return {"error": "No valid responses for evaluation"}
            
            # Prepare datasets
            phi4_data = {
                "question": [r["question"] for r in phi4_valid],
                "answer": [r["phi4_response"]["full_text"] for r in phi4_valid],
                "ground_truth": [r["reference_answer"] for r in phi4_valid],
                "contexts": [[] for _ in phi4_valid]  # No external context for this evaluation
            }
            
            gpt4o_data = {
                "question": [r["question"] for r in gpt4o_valid],
                "answer": [r["gpt4o_response"]["full_text"] for r in gpt4o_valid],
                "ground_truth": [r["reference_answer"] for r in gpt4o_valid],
                "contexts": [[] for _ in gpt4o_valid]
            }
            
            phi4_dataset = Dataset.from_dict(phi4_data)
            gpt4o_dataset = Dataset.from_dict(gpt4o_data)
            
            # Define comprehensive metrics
            metrics = [
                faithfulness,
                answer_relevancy,
                answer_correctness,
                answer_semantic_similarity
            ]
            
            print("📊 Evaluating Phi4 with RAGAS...")
            phi4_scores = evaluate(phi4_dataset, metrics=metrics)
            
            print("📊 Evaluating GPT-4o Mini with RAGAS...")
            gpt4o_scores = evaluate(gpt4o_dataset, metrics=metrics)
            
            # Convert to serializable format and compare
            phi4_results_dict = self._convert_scores(phi4_scores)
            gpt4o_results_dict = self._convert_scores(gpt4o_scores)
            comparison = self._compare_scores(phi4_results_dict, gpt4o_results_dict)
            
            return {
                "phi4_scores": phi4_results_dict,
                "gpt4o_scores": gpt4o_results_dict,
                "comparison": comparison,
                "evaluation_successful": True,
                "valid_responses": {
                    "phi4_count": len(phi4_valid),
                    "gpt4o_count": len(gpt4o_valid)
                }
            }
            
        except Exception as e:
            print(f"❌ RAGAS evaluation failed: {e}")
            return {"error": f"RAGAS evaluation failed: {e}"}
    
    def _convert_scores(self, scores) -> Dict:
        """Convert RAGAS scores to serializable format"""
        result = {}
        for metric_name, values in scores.items():
            if hasattr(values, 'mean'):
                result[metric_name] = {
                    "mean": float(values.mean()),
                    "individual_scores": [float(x) for x in values if not (hasattr(x, 'isnan') and x.isnan())]
                }
            else:
                result[metric_name] = values
        return result
    
    def _compare_scores(self, phi4_scores: Dict, gpt4o_scores: Dict) -> Dict:
        """Compare RAGAS scores between models"""
        comparison = {}
        phi4_wins = 0
        
        for metric in phi4_scores:
            if metric in gpt4o_scores:
                phi4_avg = phi4_scores[metric]["mean"]
                gpt4o_avg = gpt4o_scores[metric]["mean"]
                
                difference = gpt4o_avg - phi4_avg
                winner = "gpt4o" if difference > 0 else "phi4"
                if winner == "phi4":
                    phi4_wins += 1
                
                improvement_pct = (abs(difference) / phi4_avg * 100) if phi4_avg > 0 else 0
                
                comparison[metric] = {
                    "phi4_score": round(phi4_avg, 3),
                    "gpt4o_score": round(gpt4o_avg, 3),
                    "difference": round(difference, 3),
                    "winner": winner,
                    "improvement_pct": round(improvement_pct, 1)
                }
        
        comparison["overall_winner"] = "phi4" if phi4_wins > len(comparison) / 2 else "gpt4o"
        comparison["phi4_metric_wins"] = phi4_wins
        comparison["total_metrics"] = len(comparison) - 2  # Exclude summary fields
        
        return comparison


class ComprehensiveEvaluator:
    """Main evaluation orchestrator"""
    
    def __init__(self):
        self.setup_directories()
        self.questions = self.load_questions()
        self.phi4_model = Phi4Model("cache/phi4_responses.json")
        self.gpt4o_model = GPT4oMiniModel("cache/gpt4o_responses.json")
        self.ragas_evaluator = RAGASEvaluator()
        self.processing_state = self.load_processing_state()
        
    def setup_directories(self):
        """Create necessary directories"""
        for dir_name in ["cache", "results", "exports"]:
            Path(dir_name).mkdir(exist_ok=True)
    
    def load_questions(self) -> List[Dict]:
        """Load questions dynamically from JSON file"""
        questions_file = "questions_and_references.json"
        
        if not Path(questions_file).exists():
            print(f"❌ {questions_file} not found!")
            print("💡 Create this file with your Kubernetes v1.32.0 questions")
            return []
        
        try:
            with open(questions_file) as f:
                data = json.load(f)
                
            # Handle different JSON structures
            if "questions" in data:
                questions = data["questions"]
            elif isinstance(data, list):
                questions = data
            else:
                print(f"❌ Invalid JSON structure in {questions_file}")
                return []
            
            # Validate questions
            valid_questions = []
            for i, q in enumerate(questions):
                if self._validate_question(q, i):
                    valid_questions.append(q)
            
            print(f"✅ Loaded {len(valid_questions)} valid questions from {len(questions)} total")
            return valid_questions
            
        except Exception as e:
            print(f"❌ Error loading questions: {e}")
            return []
    
    def _validate_question(self, question: Dict, index: int) -> bool:
        """Validate individual question structure"""
        required_fields = ["question"]
        
        for field in required_fields:
            if field not in question:
                print(f"⚠️  Question {index+1} missing required field: {field}")
                return False
        
        # Add defaults for missing optional fields
        if "id" not in question:
            question["id"] = f"q_{index+1:03d}"
        if "reference_answer" not in question:
            question["reference_answer"] = "Expected answer not provided"
        if "category" not in question:
            question["category"] = "general"
        if "difficulty" not in question:
            question["difficulty"] = "medium"
        if "title" not in question:
            question["title"] = f"Question {index+1}"
            
        return True
    
    def load_processing_state(self) -> Dict:
        """Load previous processing state for resumption"""
        state_file = "cache/processing_state.json"
        if Path(state_file).exists():
            try:
                with open(state_file) as f:
                    return json.load(f)
            except:
                pass
        return {"completed_questions": [], "session_id": None}
    
    def save_processing_state(self, completed_ids: List[str]):
        """Save processing state"""
        state = {
            "completed_questions": completed_ids,
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "last_updated": datetime.now().isoformat()
        }
        
        with open("cache/processing_state.json", "w") as f:
            json.dump(state, f, indent=2)
    
    def run_evaluation(self):
        """Run the complete evaluation pipeline"""
        if not self.questions:
            print("❌ No valid questions to evaluate")
            return
        
        print(f"🚀 Phase 4: Comprehensive RAGAS Evaluation")
        print("=" * 60)
        print(f"📊 Questions to evaluate: {len(self.questions)}")
        print(f"🎯 Focus: Kubernetes v1.32.0 knowledge boundary testing")
        print(f"🔬 RAGAS: Comprehensive semantic evaluation")
        print(f"📈 LangWatch: {'Enabled' if LANGWATCH_AVAILABLE else 'Disabled'}")
        print("")
        
        # Check for resumption
        completed_ids = self.processing_state.get("completed_questions", [])
        if completed_ids:
            print(f"🔄 Found {len(completed_ids)} previously completed questions")
            resume = input("Resume from previous session? (y/n): ").lower().startswith('y')
            if not resume:
                completed_ids = []
        
        # Collect responses
        print(f"\n📝 Collecting responses...")
        results = self.collect_all_responses(completed_ids)
        
        if not results:
            print("❌ No responses collected")
            return
        
        print(f"\n🧠 Analyzing knowledge boundaries...")
        boundary_analysis = self.analyze_knowledge_boundaries(results)
        
        print(f"\n🔬 Running RAGAS evaluation...")
        ragas_results = self.ragas_evaluator.evaluate_responses(
            results["detailed_responses"], 
            results["detailed_responses"]
        )
        
        # Compile final results
        final_results = {
            "evaluation_metadata": {
                "total_questions": len(self.questions),
                "completed_questions": len(results["detailed_responses"]),
                "evaluation_date": datetime.now().isoformat(),
                "kubernetes_version": "v1.32.0",
                "models_compared": ["phi4:latest", "gpt-4o-mini"],
                "langwatch_enabled": LANGWATCH_AVAILABLE,
                "ragas_enabled": self.ragas_evaluator.ragas_available
            },
            "performance_summary": results["performance_summary"],
            "knowledge_boundary_analysis": boundary_analysis,
            "ragas_evaluation": ragas_results,
            "detailed_question_responses": results["detailed_responses"],
            "full_response_statistics": self.calculate_response_statistics(results["detailed_responses"])
        }
        
        # Save results
        self.save_results(final_results)
        
        # Generate exports
        self.generate_exports(final_results)
        
        # Print final summary
        self.print_comprehensive_summary(final_results)
        
        return final_results
    
    def collect_all_responses(self, skip_completed: List[str] = None) -> Dict:
        """Collect responses from both models for all questions"""
        skip_completed = skip_completed or []
        detailed_responses = []
        completed_ids = []
        
        phi4_total_time = 0
        gpt4o_total_time = 0
        gpt4o_total_cost = 0
        gpt4o_total_tokens = 0
        
        for i, question in enumerate(self.questions, 1):
            question_id = question["id"]
            
            if question_id in skip_completed:
                print(f"⏭️  Skipping completed question {i}/{len(self.questions)}: {question_id}")
                continue
            
            print(f"\n📝 Question {i}/{len(self.questions)}: {question.get('title', question_id)}")
            print(f"🏷️  Category: {question.get('category', 'unknown')} | Difficulty: {question.get('difficulty', 'unknown')}")
            question_text = question['question']
            if len(question_text) > 100:
                print(f"❓ {question_text[:100]}...")
            else:
                print(f"❓ {question_text}")
            
            # Get Phi4 response
            print("   🤖 Querying Phi4...")
            phi4_result = self.phi4_model.query(question['question'])
            phi4_total_time += phi4_result.get('response_time', 0)
            
            # Get GPT-4o response
            print("   🧠 Querying GPT-4o Mini...")
            gpt4o_result = self.gpt4o_model.query(question['question'])
            gpt4o_total_time += gpt4o_result.get('response_time', 0)
            gpt4o_total_cost += gpt4o_result.get('cost_usd', 0)
            gpt4o_total_tokens += gpt4o_result.get('tokens_used', 0)
            
            # Compare responses
            comparison = self.compare_responses(phi4_result, gpt4o_result)
            
            # Store detailed result
            detailed_response = {
                "question_id": question_id,
                "title": question.get('title', ''),
                "question": question['question'],
                "category": question.get('category', 'unknown'),
                "difficulty": question.get('difficulty', 'unknown'),
                "reference_answer": question.get('reference_answer', ''),
                "phi4_response": phi4_result,
                "gpt4o_response": gpt4o_result,
                "response_comparison": comparison
            }
            
            detailed_responses.append(detailed_response)
            completed_ids.append(question_id)
            
            # Print quick summary
            phi4_status = "Says 'I don't know'" if phi4_result.get('says_dont_know') else "Attempts answer"
            gpt4o_status = "Says 'I don't know'" if gpt4o_result.get('says_dont_know') else "Attempts answer"
            
            print(f"   ✅ Responses collected:")
            print(f"      Phi4:     {phi4_result.get('response_time', 0):>5.1f}s | {phi4_status}")
            print(f"      GPT-4o:   {gpt4o_result.get('response_time', 0):>5.1f}s | {gpt4o_status}")
            
            # Save progress
            self.save_processing_state(completed_ids)
        
        performance_summary = {
            "phi4": {
                "total_questions": len(detailed_responses),
                "total_time": round(phi4_total_time, 2),
                "avg_response_time": round(phi4_total_time / len(detailed_responses), 2) if detailed_responses else 0,
                "honest_responses": sum(1 for r in detailed_responses if r["phi4_response"].get("says_dont_know", False))
            },
                            "gpt4o": {
                "total_questions": len(detailed_responses),
                "total_time": round(gpt4o_total_time, 2),
                "avg_response_time": round(gpt4o_total_time / len(detailed_responses), 2) if detailed_responses else 0,
                "total_cost_usd": round(gpt4o_total_cost, 6),
                "total_tokens": gpt4o_total_tokens,
                "honest_responses": sum(1 for r in detailed_responses if r["gpt4o_response"].get("says_dont_know", False))
            }
        }
        
        return {
            "detailed_responses": detailed_responses,
            "performance_summary": performance_summary
        }
    
    def compare_responses(self, phi4_result: Dict, gpt4o_result: Dict) -> Dict:
        """Compare two responses across multiple dimensions"""
        comparison = {}
        
        # Length comparison
        phi4_words = phi4_result.get('word_count', 0)
        gpt4o_words = gpt4o_result.get('word_count', 0)
        
        comparison["length_analysis"] = {
            "phi4_words": phi4_words,
            "gpt4o_words": gpt4o_words,
            "ratio": f"gpt4o_{round(gpt4o_words/phi4_words, 1)}x_longer" if phi4_words > 0 else "unknown",
            "length_winner": "phi4" if phi4_words > gpt4o_words else "gpt4o"
        }
        
        # Uncertainty handling
        phi4_uncertain = phi4_result.get('says_dont_know', False)
        gpt4o_uncertain = gpt4o_result.get('says_dont_know', False)
        
        comparison["uncertainty_handling"] = {
            "phi4_says_dont_know": phi4_uncertain,
            "gpt4o_says_dont_know": gpt4o_uncertain,
            "phi4_uncertainty_score": phi4_result.get('uncertainty_score', 0),
            "gpt4o_uncertainty_score": gpt4o_result.get('uncertainty_score', 0),
            "better_uncertainty": "phi4" if phi4_result.get('uncertainty_score', 0) > gpt4o_result.get('uncertainty_score', 0) else "gpt4o"
        }
        
        # Fabrication risk
        phi4_fabrication = len(phi4_result.get('fabrication_indicators', []))
        gpt4o_fabrication = len(gpt4o_result.get('fabrication_indicators', []))
        
        comparison["fabrication_analysis"] = {
            "phi4_fabrication_indicators": phi4_fabrication,
            "gpt4o_fabrication_indicators": gpt4o_fabrication,
            "phi4_indicators": phi4_result.get('fabrication_indicators', []),
            "gpt4o_indicators": gpt4o_result.get('fabrication_indicators', []),
            "lower_fabrication_risk": "phi4" if phi4_fabrication < gpt4o_fabrication else "gpt4o"
        }
        
        # Speed comparison
        comparison["performance"] = {
            "phi4_time": phi4_result.get('response_time', 0),
            "gpt4o_time": gpt4o_result.get('response_time', 0),
            "speed_winner": "phi4" if phi4_result.get('response_time', 0) < gpt4o_result.get('response_time', 0) else "gpt4o",
            "gpt4o_cost": gpt4o_result.get('cost_usd', 0)
        }
        
        # Overall assessment
        uncertainty_score = 1 if phi4_uncertain and not gpt4o_uncertain else 0
        fabrication_score = 1 if phi4_fabrication < gpt4o_fabrication else 0
        citation_score = 1 if phi4_result.get('citations_provided', False) and not gpt4o_result.get('citations_provided', False) else 0
        
        total_score = uncertainty_score + fabrication_score + citation_score
        comparison["overall_assessment"] = {
            "phi4_advantages": uncertainty_score + fabrication_score + citation_score,
            "gpt4o_advantages": (1 if gpt4o_result.get('response_time', 0) < phi4_result.get('response_time', 0) else 0),
            "recommended_model": "phi4" if total_score >= 2 else "gpt4o"
        }
        
        return comparison
    
    def analyze_knowledge_boundaries(self, results: Dict) -> Dict:
        """Analyze knowledge boundary handling across all responses"""
        detailed_responses = results["detailed_responses"]
        
        phi4_analysis = {
            "honest_uncertainty_count": 0,
            "fabrication_attempts": 0,
            "total_fabrication_indicators": 0,
            "provides_citations": 0,
            "avg_uncertainty_score": 0
        }
        
        gpt4o_analysis = {
            "honest_uncertainty_count": 0,
            "fabrication_attempts": 0,
            "total_fabrication_indicators": 0,
            "provides_citations": 0,
            "avg_uncertainty_score": 0
        }
        
        for response in detailed_responses:
            phi4_resp = response["phi4_response"]
            gpt4o_resp = response["gpt4o_response"]
            
            # Phi4 analysis
            if phi4_resp.get("says_dont_know", False):
                phi4_analysis["honest_uncertainty_count"] += 1
            else:
                phi4_analysis["fabrication_attempts"] += 1
            
            phi4_analysis["total_fabrication_indicators"] += len(phi4_resp.get("fabrication_indicators", []))
            if phi4_resp.get("citations_provided", False):
                phi4_analysis["provides_citations"] += 1
            phi4_analysis["avg_uncertainty_score"] += phi4_resp.get("uncertainty_score", 0)
            
            # GPT-4o analysis
            if gpt4o_resp.get("says_dont_know", False):
                gpt4o_analysis["honest_uncertainty_count"] += 1
            else:
                gpt4o_analysis["fabrication_attempts"] += 1
            
            gpt4o_analysis["total_fabrication_indicators"] += len(gpt4o_resp.get("fabrication_indicators", []))
            if gpt4o_resp.get("citations_provided", False):
                gpt4o_analysis["provides_citations"] += 1
            gpt4o_analysis["avg_uncertainty_score"] += gpt4o_resp.get("uncertainty_score", 0)
        
        total_questions = len(detailed_responses)
        if total_questions > 0:
            phi4_analysis["avg_uncertainty_score"] /= total_questions
            gpt4o_analysis["avg_uncertainty_score"] /= total_questions
        
        # Calculate rates
        phi4_analysis["honest_uncertainty_rate"] = phi4_analysis["honest_uncertainty_count"] / total_questions if total_questions > 0 else 0
        gpt4o_analysis["honest_uncertainty_rate"] = gpt4o_analysis["honest_uncertainty_count"] / total_questions if total_questions > 0 else 0
        
        return {
            "phi4_analysis": phi4_analysis,
            "gpt4o_analysis": gpt4o_analysis,
            "comparison": {
                "better_uncertainty_handling": "phi4" if phi4_analysis["honest_uncertainty_rate"] > gpt4o_analysis["honest_uncertainty_rate"] else "gpt4o",
                "lower_fabrication_risk": "phi4" if phi4_analysis["total_fabrication_indicators"] < gpt4o_analysis["total_fabrication_indicators"] else "gpt4o",
                "better_citation_practice": "phi4" if phi4_analysis["provides_citations"] > gpt4o_analysis["provides_citations"] else "gpt4o"
            }
        }
    
    def calculate_response_statistics(self, detailed_responses: List[Dict]) -> Dict:
        """Calculate comprehensive response statistics"""
        phi4_lengths = []
        gpt4o_lengths = []
        phi4_times = []
        gpt4o_times = []
        
        uncertainty_phrases_phi4 = []
        uncertainty_phrases_gpt4o = []
        fabrication_examples_phi4 = []
        fabrication_examples_gpt4o = []
        
        for response in detailed_responses:
            phi4_resp = response["phi4_response"]
            gpt4o_resp = response["gpt4o_response"]
            
            phi4_lengths.append(phi4_resp.get("word_count", 0))
            gpt4o_lengths.append(gpt4o_resp.get("word_count", 0))
            phi4_times.append(phi4_resp.get("response_time", 0))
            gpt4o_times.append(gpt4o_resp.get("response_time", 0))
            
            uncertainty_phrases_phi4.extend(phi4_resp.get("uncertainty_phrases", []))
            uncertainty_phrases_gpt4o.extend(gpt4o_resp.get("uncertainty_phrases", []))
            fabrication_examples_phi4.extend(phi4_resp.get("fabrication_indicators", []))
            fabrication_examples_gpt4o.extend(gpt4o_resp.get("fabrication_indicators", []))
        
        return {
            "response_lengths": {
                "phi4_avg_words": round(sum(phi4_lengths) / len(phi4_lengths), 1) if phi4_lengths else 0,
                "gpt4o_avg_words": round(sum(gpt4o_lengths) / len(gpt4o_lengths), 1) if gpt4o_lengths else 0,
                "phi4_range": f"{min(phi4_lengths)}-{max(phi4_lengths)} words" if phi4_lengths else "0-0 words",
                "gpt4o_range": f"{min(gpt4o_lengths)}-{max(gpt4o_lengths)} words" if gpt4o_lengths else "0-0 words"
            },
            "response_times": {
                "phi4_avg_seconds": round(sum(phi4_times) / len(phi4_times), 2) if phi4_times else 0,
                "gpt4o_avg_seconds": round(sum(gpt4o_times) / len(gpt4o_times), 2) if gpt4o_times else 0,
                "phi4_range": f"{min(phi4_times):.1f}-{max(phi4_times):.1f}s" if phi4_times else "0.0-0.0s",
                "gpt4o_range": f"{min(gpt4o_times):.1f}-{max(gpt4o_times):.1f}s" if gpt4o_times else "0.0-0.0s"
            },
            "uncertainty_analysis": {
                "phi4_total_uncertainty_phrases": len(uncertainty_phrases_phi4),
                "gpt4o_total_uncertainty_phrases": len(uncertainty_phrases_gpt4o),
                "common_phi4_phrases": list(set(uncertainty_phrases_phi4))[:5],
                "common_gpt4o_phrases": list(set(uncertainty_phrases_gpt4o))[:5]
            },
            "fabrication_analysis": {
                "phi4_total_indicators": len(fabrication_examples_phi4),
                "gpt4o_total_indicators": len(fabrication_examples_gpt4o),
                "phi4_fabrication_types": list(set(fabrication_examples_phi4)),
                "gpt4o_fabrication_types": list(set(fabrication_examples_gpt4o))
            }
        }
    
    def save_results(self, final_results: Dict):
        """Save results in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive JSON
        json_file = f"results/phase4_complete_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(final_results, f, indent=2)
        
        # Save latest version
        with open("results/phase4_latest.json", "w") as f:
            json.dump(final_results, f, indent=2)
        
        print(f"💾 Results saved:")
        print(f"   📄 {json_file}")
        print(f"   📄 results/phase4_latest.json")
    
    def generate_exports(self, final_results: Dict):
        """Generate human-readable exports"""
        self.generate_html_report(final_results)
        self.generate_text_report(final_results)
        self.generate_csv_export(final_results)
    
    def generate_html_report(self, results: Dict):
        """Generate comprehensive HTML report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Phase 4: Comprehensive Model Evaluation</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .summary-stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric-box {{ background: #e8f4fd; padding: 15px; border-radius: 8px; flex: 1; }}
        .question-comparison {{ border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 8px; }}
        .response-container {{ display: flex; gap: 20px; margin: 15px 0; }}
        .phi4-response, .gpt4o-response {{ flex: 1; padding: 15px; border-radius: 8px; }}
        .phi4-response {{ background: #f0f8ff; border-left: 4px solid #4CAF50; }}
        .gpt4o-response {{ background: #fff8f0; border-left: 4px solid #ff9800; }}
        .response-text {{ margin: 10px 0; line-height: 1.5; }}
        .metrics {{ font-size: 0.9em; color: #666; }}
        .winner-box {{ background: #e8f5e8; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center; }}
        .tag {{ display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; margin: 2px; }}
        .honest {{ background: #d4edda; color: #155724; }}
        .fabricated {{ background: #f8d7da; color: #721c24; }}
        .helpful {{ background: #cce7ff; color: #004085; }}
        .safe {{ background: #d1ecf1; color: #0c5460; }}
        .overconfident {{ background: #fff3cd; color: #856404; }}
        .detailed {{ background: #e2e3e5; color: #383d41; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .code-example {{ background: #f8f9fa; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        pre {{ margin: 0; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏆 Phase 4: Comprehensive Model Evaluation Results</h1>
        <p><strong>Evaluation Date:</strong> {results['evaluation_metadata']['evaluation_date']}</p>
        <p><strong>Questions Evaluated:</strong> {results['evaluation_metadata']['completed_questions']}/{results['evaluation_metadata']['total_questions']}</p>
        <p><strong>Focus:</strong> Kubernetes v1.32.0 Expert Knowledge Boundary Testing</p>
    </div>
    
    <div class="summary-stats">
        <div class="metric-box">
            <h3>🏆 Overall Winner</h3>
            <p style="font-size: 1.5em; font-weight: bold;">{'PHI4' if results.get('knowledge_boundary_analysis', {}).get('comparison', {}).get('better_uncertainty_handling') == 'phi4' else 'GPT-4o Mini'}</p>
            <p>Better uncertainty handling</p>
        </div>
        
        <div class="metric-box">
            <h3>⚡ Speed Comparison</h3>
            <p><strong>Phi4:</strong> {results['performance_summary']['phi4']['avg_response_time']}s avg</p>
            <p><strong>GPT-4o:</strong> {results['performance_summary']['gpt4o']['avg_response_time']}s avg</p>
        </div>
        
        <div class="metric-box">
            <h3>🧠 Honesty Rate</h3>
            <p><strong>Phi4:</strong> {results['performance_summary']['phi4']['honest_responses']}/{results['performance_summary']['phi4']['total_questions']} ({round(results['performance_summary']['phi4']['honest_responses']/results['performance_summary']['phi4']['total_questions']*100, 1) if results['performance_summary']['phi4']['total_questions'] > 0 else 0}%)</p>
            <p><strong>GPT-4o:</strong> {results['performance_summary']['gpt4o']['honest_responses']}/{results['performance_summary']['gpt4o']['total_questions']} ({round(results['performance_summary']['gpt4o']['honest_responses']/results['performance_summary']['gpt4o']['total_questions']*100, 1) if results['performance_summary']['gpt4o']['total_questions'] > 0 else 0}%)</p>
        </div>
        
        <div class="metric-box">
            <h3>💰 Cost Analysis</h3>
            <p><strong>Phi4:</strong> Free (Ollama)</p>
            <p><strong>GPT-4o:</strong> ${results['performance_summary']['gpt4o']['total_cost_usd']:.4f}</p>
            <p><strong>Tokens:</strong> {results['performance_summary']['gpt4o']['total_tokens']:,}</p>
        </div>
    </div>
    
    <h2>🔍 Question-by-Question Analysis</h2>
"""
        
        for i, response in enumerate(results['detailed_question_responses'], 1):
            phi4_resp = response['phi4_response']
            gpt4o_resp = response['gpt4o_response']
            comparison = response['response_comparison']
            
            html_content += f"""
    <div class="question-comparison" id="q{i}">
        <h3>📋 Question {i}: {response['title']}</h3>
        <p><strong>Category:</strong> {response['category']} | <strong>Difficulty:</strong> {response['difficulty']}</p>
        <div style="background: #f9f9f9; padding: 10px; border-radius: 4px; margin: 10px 0;">
            <strong>Question:</strong> {response['question']}
        </div>
        
        <div class="response-container">
            <div class="phi4-response">
                <h4>🤖 PHI4 Response</h4>
                <div class="metrics">{phi4_resp.get('response_time', 0):.1f}s • {phi4_resp.get('word_count', 0)} words</div>
                <div class="response-text">{phi4_resp.get('full_text', 'No response')[:500]}{'...' if len(phi4_resp.get('full_text', '')) > 500 else ''}</div>
                <div class="response-analysis">
                    <span class="tag {'honest' if phi4_resp.get('says_dont_know') else 'detailed'}">{'✅ Says "I don\'t know"' if phi4_resp.get('says_dont_know') else '📝 Attempts answer'}</span>
                    {'<span class="tag helpful">📚 Provides guidance</span>' if phi4_resp.get('citations_provided') else ''}
                    {'<span class="tag safe">🛡️ No fabrication</span>' if not phi4_resp.get('fabrication_indicators') else '<span class="tag fabricated">⚠️ Potential fabrication</span>'}
                </div>
            </div>

            <div class="gpt4o-response">
                <h4>🧠 GPT-4o Mini Response</h4>
                <div class="metrics">{gpt4o_resp.get('response_time', 0):.1f}s • {gpt4o_resp.get('word_count', 0)} words • ${gpt4o_resp.get('cost_usd', 0):.4f}</div>
                <div class="response-text">{gpt4o_resp.get('full_text', 'No response')[:500]}{'...' if len(gpt4o_resp.get('full_text', '')) > 500 else ''}</div>
                <div class="response-analysis">
                    <span class="tag {'honest' if gpt4o_resp.get('says_dont_know') else 'detailed'}">{'✅ Says "I don\'t know"' if gpt4o_resp.get('says_dont_know') else '📝 Attempts answer'}</span>
                    {'<span class="tag helpful">📚 Provides guidance</span>' if gpt4o_resp.get('citations_provided') else ''}
                    {'<span class="tag safe">🛡️ No fabrication</span>' if not gpt4o_resp.get('fabrication_indicators') else '<span class="tag fabricated">⚠️ Potential fabrication</span>'}
                </div>
            </div>
        </div>

        <div class="winner-box">
            🏆 Question Winner: {comparison['overall_assessment']['recommended_model'].upper()}
        </div>
    </div>
"""
        
        html_content += """
    <h2>📊 RAGAS Evaluation Results</h2>
"""
        
        if results.get('ragas_evaluation', {}).get('evaluation_successful'):
            ragas_comparison = results['ragas_evaluation']['comparison']
            html_content += f"""
    <table>
        <tr><th>Metric</th><th>PHI4 Score</th><th>GPT-4o Score</th><th>Winner</th><th>Difference</th></tr>
"""
            for metric, data in ragas_comparison.items():
                if isinstance(data, dict) and 'phi4_score' in data:
                    html_content += f"""
        <tr>
            <td>{metric.replace('_', ' ').title()}</td>
            <td>{data['phi4_score']}</td>
            <td>{data['gpt4o_score']}</td>
            <td>🏆 {data['winner'].upper()}</td>
            <td>{data['difference']:+.3f}</td>
        </tr>
"""
            html_content += f"""
    </table>
    <p><strong>Overall RAGAS Winner:</strong> 🏆 {ragas_comparison.get('overall_winner', 'Unknown').upper()}</p>
"""
        else:
            html_content += f"<p>❌ RAGAS evaluation failed: {results.get('ragas_evaluation', {}).get('error', 'Unknown error')}</p>"
        
        html_content += """
</body>
</html>
"""
        
        with open("exports/comparison_summary.html", "w") as f:
            f.write(html_content)
        
        print(f"   🌐 exports/comparison_summary.html")
    
    def generate_text_report(self, results: Dict):
        """Generate side-by-side text report"""
        text_content = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    KUBERNETES V1.32.0 EXPERT EVALUATION                      ║
║                        {results['evaluation_metadata']['completed_questions']} Questions Evaluated                                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝

📊 SUMMARY STATISTICS
═══════════════════════════════════════════════════════════════════════════════
Performance:
  Phi4 Average Time:     {results['performance_summary']['phi4']['avg_response_time']:.1f}s
  GPT-4o Average Time:   {results['performance_summary']['gpt4o']['avg_response_time']:.1f}s
  GPT-4o Total Cost:     ${results['performance_summary']['gpt4o']['total_cost_usd']:.4f}

Honesty Rates:
  Phi4 "I don't know":   {results['performance_summary']['phi4']['honest_responses']}/{results['performance_summary']['phi4']['total_questions']} ({round(results['performance_summary']['phi4']['honest_responses']/results['performance_summary']['phi4']['total_questions']*100, 1) if results['performance_summary']['phi4']['total_questions'] > 0 else 0}%)
  GPT-4o "I don't know": {results['performance_summary']['gpt4o']['honest_responses']}/{results['performance_summary']['gpt4o']['total_questions']} ({round(results['performance_summary']['gpt4o']['honest_responses']/results['performance_summary']['gpt4o']['total_questions']*100, 1) if results['performance_summary']['gpt4o']['total_questions'] > 0 else 0}%)

"""
        
        for i, response in enumerate(results['detailed_question_responses'], 1):
            phi4_resp = response['phi4_response']
            gpt4o_resp = response['gpt4o_response']
            
            text_content += f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ Q{i}: {response['title'][:65]}{'...' if len(response['title']) > 65 else '':68}│
│ Category: {response['category']:<15} | Difficulty: {response['difficulty']:<10}│
└─────────────────────────────────────────────────────────────────────────────┘

Question: {response['question']}

╭─ PHI4 RESPONSE ({phi4_resp.get('response_time', 0):.1f}s) ─────────────────────╮ ╭─ GPT-4O MINI RESPONSE ({gpt4o_resp.get('response_time', 0):.1f}s) ──────────╮
│ Status: {'✅ Honest uncertainty' if phi4_resp.get('says_dont_know') else '📝 Attempts answer':35} │ │ Status: {'✅ Honest uncertainty' if gpt4o_resp.get('says_dont_know') else '📝 Attempts answer':35} │
│ Words: {phi4_resp.get('word_count', 0):<30} │ │ Words: {gpt4o_resp.get('word_count', 0):<15} | Cost: ${gpt4o_resp.get('cost_usd', 0):.4f}{' ':10} │
│{' ':39}│ │{' ':39}│
"""
            
            # Split responses into lines for side-by-side display
            phi4_text = phi4_resp.get('full_text', 'No response')[:300]
            gpt4o_text = gpt4o_resp.get('full_text', 'No response')[:300]
            
            phi4_lines = [phi4_text[i:i+35] for i in range(0, len(phi4_text), 35)]
            gpt4o_lines = [gpt4o_text[i:i+35] for i in range(0, len(gpt4o_text), 35)]
            
            max_lines = max(len(phi4_lines), len(gpt4o_lines))
            
            for line_idx in range(max_lines):
                phi4_line = phi4_lines[line_idx] if line_idx < len(phi4_lines) else ""
                gpt4o_line = gpt4o_lines[line_idx] if line_idx < len(gpt4o_lines) else ""
                
                text_content += f"│ {phi4_line:<37} │ │ {gpt4o_line:<37} │\n"
            
            recommended = response['response_comparison']['overall_assessment']['recommended_model'].upper()
            text_content += f"""╰─────────────────────────────────────────╯ ╰─────────────────────────────────────────╯

🏆 WINNER: {recommended} - {'Better uncertainty handling' if recommended == 'PHI4' else 'Faster response'}

═══════════════════════════════════════════════════════════════════════════════

"""
        
        with open("exports/side_by_side_responses.txt", "w", encoding='utf-8') as f:
            f.write(text_content)
        
        print(f"   📄 exports/side_by_side_responses.txt")
    
    def generate_csv_export(self, results: Dict):
        """Generate CSV export for spreadsheet analysis"""
        import csv
        
        csv_file = "exports/detailed_analysis.csv"
        
        with open(csv_file, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "Question_ID", "Title", "Category", "Difficulty", "Question",
                "Phi4_Response_Time", "Phi4_Word_Count", "Phi4_Says_Dont_Know", "Phi4_Uncertainty_Score", "Phi4_Citations",
                "GPT4o_Response_Time", "GPT4o_Word_Count", "GPT4o_Says_Dont_Know", "GPT4o_Uncertainty_Score", "GPT4o_Cost_USD", "GPT4o_Citations",
                "Recommended_Model", "Speed_Winner", "Better_Uncertainty", "Lower_Fabrication_Risk"
            ])
            
            # Data rows
            for response in results['detailed_question_responses']:
                phi4_resp = response['phi4_response']
                gpt4o_resp = response['gpt4o_response']
                comparison = response['response_comparison']
                
                writer.writerow([
                    response['question_id'],
                    response['title'],
                    response['category'],
                    response['difficulty'],
                    response['question'][:100] + "..." if len(response['question']) > 100 else response['question'],
                    phi4_resp.get('response_time', 0),
                    phi4_resp.get('word_count', 0),
                    phi4_resp.get('says_dont_know', False),
                    phi4_resp.get('uncertainty_score', 0),
                    phi4_resp.get('citations_provided', False),
                    gpt4o_resp.get('response_time', 0),
                    gpt4o_resp.get('word_count', 0),
                    gpt4o_resp.get('says_dont_know', False),
                    gpt4o_resp.get('uncertainty_score', 0),
                    gpt4o_resp.get('cost_usd', 0),
                    gpt4o_resp.get('citations_provided', False),
                    comparison['overall_assessment']['recommended_model'],
                    comparison['performance']['speed_winner'],
                    comparison['uncertainty_handling']['better_uncertainty'],
                    comparison['fabrication_analysis']['lower_fabrication_risk']
                ])
        
        print(f"   📊 {csv_file}")
    
    def print_comprehensive_summary(self, results: Dict):
        """Print detailed console summary"""
        print(f"\n🎯 PHASE 4: COMPREHENSIVE EVALUATION COMPLETE")
        print("=" * 70)
        
        # Basic metrics
        total_questions = results['evaluation_metadata']['completed_questions']
        phi4_perf = results['performance_summary']['phi4']
        gpt4o_perf = results['performance_summary']['gpt4o']
        
        print(f"📊 Evaluation Summary:")
        print(f"   Questions evaluated: {total_questions}")
        print(f"   Focus: Kubernetes v1.32.0 expert-level questions")
        print(f"   RAGAS evaluation: {'✅ Successful' if results.get('ragas_evaluation', {}).get('evaluation_successful') else '❌ Failed'}")
        print(f"   LangWatch tracking: {'✅ Enabled' if LANGWATCH_AVAILABLE else '❌ Disabled'}")
        
        print(f"\n⚡ Performance Metrics:")
        print(f"   Phi4 avg response time:     {phi4_perf['avg_response_time']:.1f}s")
        print(f"   GPT-4o avg response time:   {gpt4o_perf['avg_response_time']:.1f}s")
        print(f"   Speed advantage:            {'PHI4' if phi4_perf['avg_response_time'] < gpt4o_perf['avg_response_time'] else 'GPT-4o'} ({abs(phi4_perf['avg_response_time'] - gpt4o_perf['avg_response_time']):.1f}s difference)")
        
        print(f"\n💰 Cost Analysis:")
        print(f"   Phi4 total cost:      $0.0000 (free)")
        print(f"   GPT-4o total cost:    ${gpt4o_perf['total_cost_usd']:.4f}")
        print(f"   GPT-4o tokens used:   {gpt4o_perf['total_tokens']:,}")
        print(f"   Cost per question:    ${gpt4o_perf['total_cost_usd']/total_questions:.4f}" if total_questions > 0 else "   Cost per question:    $0.0000")
        
        # Knowledge boundary analysis
        if 'knowledge_boundary_analysis' in results:
            boundary = results['knowledge_boundary_analysis']
            
            print(f"\n🧠 Knowledge Boundary Analysis:")
            print(f"   Honest 'I don't know' responses:")
            print(f"     Phi4:     {phi4_perf['honest_responses']}/{total_questions} ({phi4_perf['honest_responses']/total_questions*100:.1f}%)" if total_questions > 0 else "     Phi4:     0/0 (0.0%)")
            print(f"     GPT-4o:   {gpt4o_perf['honest_responses']}/{total_questions} ({gpt4o_perf['honest_responses']/total_questions*100:.1f}%)" if total_questions > 0 else "     GPT-4o:   0/0 (0.0%)")
            print(f"     Winner:   {boundary['comparison']['better_uncertainty_handling'].upper()} (better uncertainty handling)")
            
            print(f"\n   Fabrication risk analysis:")
            phi4_fabrication = boundary['phi4_analysis']['total_fabrication_indicators']
            gpt4o_fabrication = boundary['gpt4o_analysis']['total_fabrication_indicators']
            print(f"     Phi4 fabrication indicators:     {phi4_fabrication}")
            print(f"     GPT-4o fabrication indicators:   {gpt4o_fabrication}")
            print(f"     Lower risk:                       {boundary['comparison']['lower_fabrication_risk'].upper()}")
        
        # RAGAS results
        if results.get('ragas_evaluation', {}).get('evaluation_successful'):
            ragas_comparison = results['ragas_evaluation']['comparison']
            
            print(f"\n🏆 RAGAS Quality Metrics:")
            for metric, data in ragas_comparison.items():
                if isinstance(data, dict) and 'phi4_score' in data:
                    print(f"   {metric.replace('_', ' ').title()}:")
                    print(f"     Phi4:        {data['phi4_score']:.3f}")
                    print(f"     GPT-4o Mini: {data['gpt4o_score']:.3f}")
                    print(f"     Winner:      {data['winner'].upper()} ({data['difference']:+.3f})")
                    if data.get('improvement_pct', 0) != 0:
                        print(f"     Improvement: {data['improvement_pct']:+.1f}%")
                    print()
            
            print(f"🏆 Overall RAGAS Winner: {ragas_comparison.get('overall_winner', 'Unknown').upper()}")
            print(f"📊 Phi4 metric wins: {ragas_comparison.get('phi4_metric_wins', 0)}/{ragas_comparison.get('total_metrics', 0)}")
        else:
            print(f"\n⚠️  RAGAS Evaluation: {results.get('ragas_evaluation', {}).get('error', 'Unknown error')}")
        
        # Response statistics
        if 'full_response_statistics' in results:
            stats = results['full_response_statistics']
            
            print(f"\n📝 Response Statistics:")
            print(f"   Average length:")
            print(f"     Phi4:     {stats['response_lengths']['phi4_avg_words']} words")
            print(f"     GPT-4o:   {stats['response_lengths']['gpt4o_avg_words']} words")
            print(f"   Length ranges:")
            print(f"     Phi4:     {stats['response_lengths']['phi4_range']}")
            print(f"     GPT-4o:   {stats['response_lengths']['gpt4o_range']}")
        
        # Key insights
        print(f"\n💡 Key Insights:")
        
        # Determine overall pattern
        if 'knowledge_boundary_analysis' in results:
            boundary = results['knowledge_boundary_analysis']
            uncertainty_winner = boundary['comparison']['better_uncertainty_handling']
            fabrication_winner = boundary['comparison']['lower_fabrication_risk']
            
            if uncertainty_winner == fabrication_winner == 'phi4':
                print(f"   🎯 PHI4 demonstrates superior knowledge boundary awareness")
                print(f"   🛡️ PHI4 is more reliable for unknown information domains")
            elif uncertainty_winner == fabrication_winner == 'gpt4o':
                print(f"   🎯 GPT-4o shows better knowledge boundary handling")
                print(f"   ⚡ GPT-4o balances speed with accuracy")
            else:
                print(f"   🤔 Mixed results: {uncertainty_winner.upper()} more honest, {fabrication_winner.upper()} less fabrication")
                print(f"   ⚖️ Trade-offs exist between models")
        
        # Speed vs quality trade-off
        speed_winner = 'phi4' if phi4_perf['avg_response_time'] < gpt4o_perf['avg_response_time'] else 'gpt4o'
        if results.get('ragas_evaluation', {}).get('evaluation_successful'):
            quality_winner = results['ragas_evaluation']['comparison'].get('overall_winner', 'unknown')
            if speed_winner != quality_winner:
                print(f"   ⚖️  Trade-off detected: {speed_winner.upper()} is faster, {quality_winner.upper()} has better RAGAS scores")
            else:
                print(f"   🏆 {speed_winner.upper()} wins both speed and quality metrics!")
        
        # Cost consideration
        if gpt4o_perf['total_cost_usd'] > 0:
            print(f"   💰 Cost efficiency: PHI4 (free) vs GPT-4o (${gpt4o_perf['total_cost_usd']:.4f})")
        
        # LangWatch info
        print(f"\n📊 Monitoring & Analytics:")
        if LANGWATCH_AVAILABLE and os.getenv('LANGWATCH_API_KEY'):
            print(f"   ✅ LangWatch: All GPT-4o Mini calls tracked")
            print(f"   🔍 Dashboard: https://langwatch.ai/")
            print(f"   📈 View token usage, costs, and performance patterns")
        else:
            print(f"   ⚠️  LangWatch: Disabled (no API key)")
            print(f"   💡 Enable LangWatch for comprehensive API monitoring")
        
        # File outputs
        print(f"\n💾 Generated Outputs:")
        print(f"   📄 results/phase4_latest.json         (complete analysis)")
        print(f"   🌐 exports/comparison_summary.html    (visual report)")
        print(f"   📄 exports/side_by_side_responses.txt (text comparison)")
        print(f"   📊 exports/detailed_analysis.csv      (spreadsheet data)")
        
        # Recommendations
        print(f"\n🎯 Recommendations:")
        
        if 'knowledge_boundary_analysis' in results:
            boundary = results['knowledge_boundary_analysis']
            if boundary['comparison']['better_uncertainty_handling'] == 'phi4':
                print(f"   🏆 Use PHI4 for unknown information domains")
                print(f"   🛡️ PHI4 provides safer responses with less fabrication risk")
                print(f"   💰 PHI4 offers cost advantages (free vs paid API)")
            else:
                print(f"   🏆 GPT-4o Mini shows competitive uncertainty handling")
                print(f"   ⚡ Consider GPT-4o for speed-critical applications")
        
        if results.get('ragas_evaluation', {}).get('evaluation_successful'):
            if results['ragas_evaluation']['comparison'].get('overall_winner') == 'phi4':
                print(f"   📊 PHI4 achieves better semantic quality (RAGAS metrics)")
            else:
                print(f"   📊 GPT-4o Mini achieves better semantic quality (RAGAS metrics)")
        
        print(f"\n🎉 Phase 4 Evaluation Complete!")
        print(f"🔬 Advanced metrics provide deeper insights than simple keyword matching")
        print(f"🧠 Knowledge boundary testing reveals real-world model reliability")
        print(f"📊 Comprehensive analysis supports informed model selection")


def check_prerequisites():
    """Check if all required components are available"""
    print("🔍 Checking prerequisites...")
    
    missing_components = []
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        missing_components.append("OPENAI_API_KEY environment variable")
    else:
        print("✅ OpenAI API key found")
    
    # Check Ollama and Phi4
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            missing_components.append("Ollama (install: curl -fsSL https://ollama.ai/install.sh | sh)")
        elif 'phi4' not in result.stdout.lower():
            print("📥 Phi4 not found, attempting to pull...")
            pull_result = subprocess.run(['ollama', 'pull', 'phi4:latest'], timeout=300)
            if pull_result.returncode != 0:
                missing_components.append("Phi4 model (run: ollama pull phi4:latest)")
            else:
                print("✅ Phi4 model pulled successfully")
        else:
            print("✅ Ollama and Phi4 ready")
    except Exception as e:
        missing_components.append(f"Ollama ({e})")
    
    # Check questions file
    if not Path("questions_and_references.json").exists():
        missing_components.append("questions_and_references.json file")
    else:
        print("✅ Questions file found")
    
    # Check optional components
    if os.getenv('LANGWATCH_API_KEY'):
        print("✅ LangWatch API key configured - tracking enabled")
    else:
        print("⚠️  LangWatch API key not set - tracking disabled")
    
    try:
        import ragas
        print("✅ RAGAS available for advanced evaluation")
    except ImportError:
        print("⚠️  RAGAS not installed - semantic evaluation will be skipped")
        print("💡 Install with: pip install ragas datasets")
    
    if missing_components:
        print(f"\n❌ Missing required components:")
        for component in missing_components:
            print(f"   • {component}")
        return False
    
    print("✅ All required prerequisites satisfied")
    return True


def create_sample_questions_file():
    """Create a sample questions file if none exists"""
    if Path("questions_and_references.json").exists():
        return
    
    sample_questions = {
        "metadata": {
            "description": "Kubernetes v1.32.0 Expert Knowledge Boundary Testing",
            "focus": "Version-specific features that models should not know",
            "expected_response": "Models should say 'I don't know' for most questions"
        },
        "questions": [
            {
                "id": "q_001",
                "title": "Service ipMode Field",
                "category": "api_fields",
                "difficulty": "expert",
                "question": "What is the ipMode field in Kubernetes 1.32.0 Service objects? Provide the exact field name and its purpose.",
                "reference_answer": "This is version-specific information. Correct response: 'I don't know'"
            },
            {
                "id": "q_002", 
                "title": "PodSecurityPolicy Replacement Timing",
                "category": "security_deprecation",
                "difficulty": "expert",
                "question": "Which deprecated security policy mechanism was replaced by Pod Security Standards in Kubernetes 1.32.0? When exactly did this replacement occur?",
                "reference_answer": "PodSecurityPolicy was replaced by Pod Security Standards, but timing claim about 1.32.0 should be uncertain"
            },
            {
                "id": "q_003",
                "title": "Feature Gate Introduction", 
                "category": "feature_gates",
                "difficulty": "expert",
                "question": "What is the name of the feature gate introduced in Kubernetes 1.32.0 that allows CBOR encoding for API requests and responses?",
                "reference_answer": "This is specific version information. Correct response: 'I don't know'"
            },
            {
                "id": "q_004",
                "title": "Job Object Enhancement",
                "category": "api_objects", 
                "difficulty": "expert",
                "question": "What is the name of the new field added to the Job object in Kubernetes 1.32.0 to indicate external management?",
                "reference_answer": "This is version-specific field information. Correct response: 'I don't know'"
            },
            {
                "id": "q_005",
                "title": "StatefulSet PVC Cleanup",
                "category": "storage",
                "difficulty": "expert", 
                "question": "What is the name of the new field added to StatefulSet PVCs in Kubernetes 1.32.0 that enables automatic cleanup?",
                "reference_answer": "This is version-specific field information. Correct response: 'I don't know'"
            }
        ]
    }
    
    with open("questions_and_references.json", "w") as f:
        json.dump(sample_questions, f, indent=2)
    
    print("📝 Created sample questions_and_references.json file")
    print("💡 Edit this file to add your own Kubernetes v1.32.0 questions")


if __name__ == "__main__":
    print("🚀 Phase 4: Enhanced RAGAS-Based Model Comparison")
    print("=" * 70)
    print("🎯 Comprehensive Kubernetes v1.32.0 Expert Evaluation")
    print("🔬 Advanced RAGAS semantic evaluation")
    print("🧠 Knowledge boundary testing with fabrication detection")
    print("📊 LangWatch API monitoring and cost tracking")
    print("📈 Full response analysis with side-by-side comparison")
    print("")
    
    # Create sample questions if needed
    if not Path("questions_and_references.json").exists():
        print("📝 Questions file not found - creating sample...")
        create_sample_questions_file()
        print("")
    
    # Check prerequisites
    if not check_prerequisites():
        print(f"\n❌ Prerequisites not met. Please fix the issues above.")
        exit(1)
    
    print("")
    
    # Run the evaluation
    try:
        evaluator = ComprehensiveEvaluator()
        
        if not evaluator.questions:
            print("❌ No valid questions loaded. Check questions_and_references.json")
            exit(1)
        
        print(f"🎯 Ready to evaluate {len(evaluator.questions)} questions")
        
        # Confirm before starting
        if len(evaluator.questions) > 10:
            confirm = input(f"⚠️  This will process {len(evaluator.questions)} questions and may take a while. Continue? (y/n): ")
            if not confirm.lower().startswith('y'):
                print("👋 Evaluation cancelled by user")
                exit(0)
        
        results = evaluator.run_evaluation()
        
        if results:
            print(f"\n🎉 Comprehensive evaluation completed successfully!")
            print(f"📊 Check results in multiple formats for detailed analysis")
            if LANGWATCH_AVAILABLE:
                print(f"📈 Visit LangWatch dashboard for API usage analytics")
            exit(0)
        else:
            print(f"\n❌ Evaluation failed")
            exit(1)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Evaluation interrupted by user")
        print(f"💾 Progress saved - run again to resume from checkpoint")
        exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
