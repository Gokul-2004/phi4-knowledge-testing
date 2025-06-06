#!/usr/bin/env python3
"""
Kubernetes 1.32.0 Evaluator - Loads questions from JSON file
Improved accuracy with focused questions and knowledge disclaimers
"""

import subprocess
import time
import json
import sys
from datetime import datetime
from pathlib import Path
import statistics

class Kubernetes132Evaluator:
    def __init__(self, questions_file="k8s_132_questions_focused.json"):
        self.results_dir = Path("results")
        self.cache_dir = Path("cache") 
        self.questions_file = Path(questions_file)
        
        # Create directories
        self.results_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load questions from JSON
        self.questions_data = self.load_questions()
        
        # Load cache if exists
        self.cache_file = self.cache_dir / "k8s_132_cache.json"
        self.cache = self.load_cache()
        
    def load_questions(self):
        """Load questions from JSON file"""
        if not self.questions_file.exists():
            print(f"❌ Questions file not found: {self.questions_file}")
            print("💡 Please create k8s_132_questions_focused.json first")
            sys.exit(1)
            
        try:
            with open(self.questions_file, 'r') as f:
                questions_data = json.load(f)
            
            print(f"📋 Loaded {questions_data['metadata']['total_questions']} questions")
            print(f"📅 Version: {questions_data['metadata']['version']}")
            print(f"🎯 Type: {questions_data['metadata']['question_type']}")
            
            return questions_data
            
        except Exception as e:
            print(f"❌ Error loading questions: {e}")
            sys.exit(1)
        
    def load_cache(self):
        """Load existing results to resume interrupted evaluations"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                print(f"📁 Loaded cache with {len(cache)} previous results")
                return cache
            except Exception as e:
                print(f"⚠️ Cache load failed: {e}")
        return {}
    
    def save_cache(self):
        """Save current results to cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"⚠️ Cache save failed: {e}")
    
    def check_ollama_phi4(self):
        """Verify Ollama and Phi4 are ready"""
        print("🔍 Checking Ollama and Phi4...")
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print("❌ Ollama not found. Install: curl -fsSL https://ollama.ai/install.sh | sh")
                return False
            
            if 'phi4' not in result.stdout.lower():
                print("💡 Installing Phi4...")
                subprocess.run(['ollama', 'pull', 'phi4:latest'], timeout=600)
            
            print("✅ Ollama and Phi4 ready!")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def query_phi4(self, question, question_id):
        """Query Phi4 with caching support"""
        # Check cache first
        if question_id in self.cache:
            print("💾 Using cached result")
            cached = self.cache[question_id]
            return cached['response'], cached['response_time']
        
        print("🤖 Querying Phi4...")
        start_time = time.time()
        try:
            result = subprocess.run(['ollama', 'run', 'phi4:latest', question], 
                                  capture_output=True, text=True, timeout=300)
            response_time = round(time.time() - start_time, 2)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                
                # Cache the result
                self.cache[question_id] = {
                    'response': response,
                    'response_time': response_time,
                    'timestamp': datetime.now().isoformat()
                }
                self.save_cache()
                
                return response, response_time
            else:
                error_msg = f"ERROR: {result.stderr}"
                return error_msg, response_time
                
        except subprocess.TimeoutExpired:
            response_time = round(time.time() - start_time, 2)
            return "ERROR: Query timeout", response_time
        except Exception as e:
            response_time = round(time.time() - start_time, 2)
            return f"ERROR: {e}", response_time
    
    def score_response_focused(self, response, keywords):
        """Enhanced scoring for focused responses"""
        response_lower = response.lower()
        
        # Check for "I don't know" - this should get high accuracy if appropriate
        if "i don't know" in response_lower or "i do not know" in response_lower:
            return {
                "accuracy": 0.9,  # High score for honest uncertainty
                "completeness": 0.8,
                "relevance": 1.0,
                "knowledge_boundary": 1.0
            }
        
        # Count keywords
        essential_hits = sum(1 for kw in keywords["essential"] if kw.lower() in response_lower)
        important_hits = sum(1 for kw in keywords["important"] if kw.lower() in response_lower)
        
        # Calculate accuracy
        accuracy = (essential_hits / len(keywords["essential"])) * 0.8 + (important_hits / len(keywords["important"])) * 0.2
        
        # Completeness - reward conciseness for focused questions
        word_count = len(response.split())
        if word_count <= 50:  # Focused response
            completeness = 1.0
        elif word_count <= 100:  # Moderate
            completeness = 0.8
        else:  # Too verbose
            completeness = max(0.3, 1.0 - (word_count - 100) / 200)
        
        # Relevance - check for version-specific terms
        version_terms = ['1.32.0', '1.32', 'v1.32']
        relevance = 1.0 if any(term in response_lower for term in version_terms) else 0.5
        
        # Knowledge boundary - penalize fabricated details for unknown info
        knowledge_boundary = 0.8 if word_count > 200 else 1.0
        
        return {
            "accuracy": round(min(1.0, accuracy), 3),
            "completeness": round(completeness, 3),
            "relevance": round(relevance, 3),
            "knowledge_boundary": round(knowledge_boundary, 3)
        }
    
    def run_evaluation(self, sample_size=None):
        """Run evaluation with focused questions"""
        print("🎯 Kubernetes 1.32.0 Focused Evaluation")
        print("=" * 50)
        
        if not self.check_ollama_phi4():
            return {}
        
        questions = self.questions_data['questions']
        if sample_size:
            questions = questions[:sample_size]
            print(f"📝 Running sample of {len(questions)} questions")
        
        results = []
        
        for i, q in enumerate(questions, 1):
            print(f"\n📝 Question {i}/{len(questions)}: {q['id']}")
            print(f"❓ {q['question']}")
            
            response, response_time = self.query_phi4(q['question'], q['id'])
            
            if response.startswith("ERROR"):
                print(f"❌ {response}")
                continue
            
            scores = self.score_response_focused(response, q['keywords'])
            word_count = len(response.split())
            
            print(f"✅ {word_count} words, {response_time}s")
            print(f"📊 Acc:{scores['accuracy']:.2f} Comp:{scores['completeness']:.2f} Rel:{scores['relevance']:.2f} KB:{scores['knowledge_boundary']:.2f}")
            
            # Show first line of response for quick check
            first_line = response.split('\n')[0][:80] + "..." if len(response.split('\n')[0]) > 80 else response.split('\n')[0]
            print(f"💬 Response: {first_line}")
            
            results.append({
                "question_id": q['id'],
                "question": q['question'],
                "response": response,
                "response_time": response_time,
                "word_count": word_count,
                "scores": scores,
                "timestamp": datetime.now().isoformat()
            })
        
        # Generate report
        if results:
            # Calculate overall score
            overall_score = statistics.mean([
                statistics.mean([r['scores']['accuracy'] for r in results]),
                statistics.mean([r['scores']['completeness'] for r in results]),
                statistics.mean([r['scores']['relevance'] for r in results]),
                statistics.mean([r['scores']['knowledge_boundary'] for r in results])
            ])
            
            report = {
                "evaluation_metadata": {
                    "model": "phi4:latest",
                    "evaluation_date": datetime.now().isoformat(),
                    "questions_version": self.questions_data['metadata']['version'],
                    "question_type": self.questions_data['metadata']['question_type'],
                    "total_questions": len(results),
                    "success_rate": len([r for r in results if not r['response'].startswith('ERROR')]) / len(results)
                },
                "performance_metrics": {
                    "avg_response_time": round(statistics.mean([r['response_time'] for r in results]), 2),
                    "avg_word_count": round(statistics.mean([r['word_count'] for r in results]), 1),
                    "responses_with_dont_know": len([r for r in results if "don't know" in r['response'].lower()])
                },
                "quality_metrics": {
                    "overall_score": round(overall_score, 3),
                    "avg_accuracy": round(statistics.mean([r['scores']['accuracy'] for r in results]), 3),
                    "avg_completeness": round(statistics.mean([r['scores']['completeness'] for r in results]), 3),
                    "avg_relevance": round(statistics.mean([r['scores']['relevance'] for r in results]), 3),
                    "avg_knowledge_boundary": round(statistics.mean([r['scores']['knowledge_boundary'] for r in results]), 3)
                },
                "detailed_results": results
            }
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"k8s_132_focused_evaluation_{timestamp}.json"
            
            with open(self.results_dir / filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Also save as latest
            with open(self.results_dir / "k8s_132_latest.json", 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n🎯 KUBERNETES 1.32.0 EVALUATION SUMMARY")
            print("=" * 45)
            print(f"📊 Questions: {len(results)}")
            print(f"🎯 Overall Score: {overall_score:.3f}/1.000")
            print(f"🎪 Accuracy: {report['quality_metrics']['avg_accuracy']:.3f}")
            print(f"📝 Avg Words: {report['performance_metrics']['avg_word_count']}")
            print(f"⚡ Avg Time: {report['performance_metrics']['avg_response_time']}s")
            print(f"🤷 'Don't know' responses: {report['performance_metrics']['responses_with_dont_know']}")
            print(f"💾 Results: {filename}")
            
            return report
        
        return {}

def main():
    """Main evaluation runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kubernetes 1.32.0 Evaluation')
    parser.add_argument('--sample', type=int, help='Run sample of N questions (default: all)')
    parser.add_argument('--questions', default='k8s_132_questions_focused.json', help='Questions JSON file')
    
    args = parser.parse_args()
    
    evaluator = Kubernetes132Evaluator(questions_file=args.questions)
    report = evaluator.run_evaluation(sample_size=args.sample)
    
    if report:
        print("\n🎉 Evaluation complete!")
        overall_score = report['quality_metrics']['overall_score']
        
        if overall_score > 0.8:
            print("🏆 Excellent performance!")
        elif overall_score > 0.6:
            print("✅ Good performance - significant improvement!")
        else:
            print("⚠️ Room for improvement")
            
        return 0
    else:
        print("❌ Evaluation failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
