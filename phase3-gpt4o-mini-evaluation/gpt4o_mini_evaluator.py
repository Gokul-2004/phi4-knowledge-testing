#!/usr/bin/env python3
"""
Phase 3: GPT-4o Mini Evaluation with Kubernetes 1.32 Knowledge Detection
Enhanced with caching, resumption, and proper sample management
"""

import openai
import time
import json
import sys
import os
from datetime import datetime
from pathlib import Path
import statistics
from typing import Dict, List, Optional, Tuple

class GPT4oMiniEvaluator:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize evaluator with OpenAI API key and setup directories"""
        Path("results").mkdir(exist_ok=True)
        Path("cache").mkdir(exist_ok=True)
        Path("samples").mkdir(exist_ok=True)
        
        # Setup OpenAI client
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            # Try environment variable
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("❌ No OpenAI API key provided!")
                print("💡 Set OPENAI_API_KEY environment variable or pass as parameter")
                sys.exit(1)
            self.client = openai.OpenAI(api_key=api_key)
        
        self.model = "gpt-4o-mini"
        self.cache_file = "cache/gpt4o_responses.json"
        self.load_cache()
    
    def load_cache(self):
        """Load cached responses to avoid re-querying"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                print(f"📂 Loaded {len(self.cache)} cached responses")
            else:
                self.cache = {}
        except Exception as e:
            print(f"⚠️ Could not load cache: {e}")
            self.cache = {}
    
    def save_cache(self):
        """Save responses to cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save cache: {e}")
    
    def get_cache_key(self, question: str) -> str:
        """Generate cache key for question"""
        import hashlib
        return hashlib.md5(question.encode()).hexdigest()[:12]
    
    def query_gpt4o_mini(self, question: str) -> Tuple[str, float]:
        """Query GPT-4o mini with caching support"""
        cache_key = self.get_cache_key(question)
        
        # Check cache first
        if cache_key in self.cache:
            print("💾 Using cached response")
            return self.cache[cache_key]['response'], self.cache[cache_key]['response_time']
        
        print("🤖 Querying GPT-4o mini...")
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant with expertise in Kubernetes. Provide detailed, accurate responses."},
                    {"role": "user", "content": question}
                ],
                max_tokens=1000,
                temperature=0.1  # Lower temperature for more consistent responses
            )
            
            response_time = round(time.time() - start_time, 2)
            response_text = response.choices[0].message.content.strip()
            
            # Cache the response
            self.cache[cache_key] = {
                'question': question,
                'response': response_text,
                'response_time': response_time,
                'timestamp': datetime.now().isoformat()
            }
            self.save_cache()
            
            return response_text, response_time
            
        except Exception as e:
            print(f"❌ Error querying GPT-4o mini: {e}")
            return f"ERROR: {e}", round(time.time() - start_time, 2)
    
    def check_k8s_132_knowledge(self) -> Dict:
        """Check if GPT-4o mini has specific Kubernetes 1.32 knowledge"""
        print("🔍 Testing Kubernetes 1.32 Specific Knowledge")
        print("=" * 50)
        
        # Two specific Kubernetes 1.32 questions
        k8s_132_questions = [
            {
                "id": "k8s_132_q1",
                "question": "What are the new features introduced in Kubernetes version 1.32? Please be specific about version 1.32 only.",
                "focus": "version_specific_features"
            },
            {
                "id": "k8s_132_q2", 
                "question": "What changes were made to the Kubernetes API in version 1.32? Include any deprecated or beta features that became stable.",
                "focus": "api_changes"
            }
        ]
        
        results = []
        
        for q in k8s_132_questions:
            print(f"\n📝 {q['id']}: {q['focus']}")
            print(f"❓ {q['question']}")
            
            response, response_time = self.query_gpt4o_mini(q['question'])
            
            if response.startswith("ERROR"):
                print(f"❌ {response}")
                continue
            
            # Analyze response for 1.32 specific knowledge
            analysis = self.analyze_k8s_132_response(response)
            word_count = len(response.split())
            
            print(f"✅ {word_count} words, {response_time}s")
            print(f"🔍 Analysis: {analysis['confidence']}")
            
            results.append({
                "question_id": q['id'],
                "question": q['question'],
                "focus": q['focus'],
                "response": response,
                "response_time": response_time,
                "word_count": word_count,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            })
            
            # Save individual sample
            self.save_sample(f"k8s_132_{q['id']}", {
                "question": q['question'],
                "response": response,
                "analysis": analysis,
                "metadata": {
                    "model": self.model,
                    "response_time": response_time,
                    "word_count": word_count,
                    "timestamp": datetime.now().isoformat()
                }
            })
        
        return {"k8s_132_detection": results}
    
    def analyze_k8s_132_response(self, response: str) -> Dict:
        """Analyze response for Kubernetes 1.32 specific knowledge"""
        response_lower = response.lower()
        
        # Check for version mentions
        has_132_mention = "1.32" in response_lower
        has_version_awareness = any(v in response_lower for v in ["1.30", "1.31", "1.32", "1.33"])
        
        # Check for recent/general indicators
        uncertainty_indicators = [
            "i don't have", "not aware", "cannot provide", "as of my last update",
            "may not be current", "check official", "latest information"
        ]
        shows_uncertainty = any(indicator in response_lower for indicator in uncertainty_indicators)
        
        # Check for confident specific claims
        specific_indicators = [
            "introduced in 1.32", "new in version 1.32", "1.32 features",
            "stable in 1.32", "deprecated in 1.32"
        ]
        makes_specific_claims = any(indicator in response_lower for indicator in specific_indicators)
        
        # Determine confidence level
        if makes_specific_claims and has_132_mention:
            confidence = "HIGH - Makes specific 1.32 claims"
        elif has_132_mention and not shows_uncertainty:
            confidence = "MEDIUM - Mentions 1.32 without uncertainty"
        elif shows_uncertainty:
            confidence = "LOW - Shows uncertainty about recent versions"
        elif has_version_awareness:
            confidence = "MEDIUM - Version aware but not 1.32 specific"
        else:
            confidence = "NONE - No version-specific knowledge"
        
        return {
            "confidence": confidence,
            "has_132_mention": has_132_mention,
            "has_version_awareness": has_version_awareness,
            "shows_uncertainty": shows_uncertainty,
            "makes_specific_claims": makes_specific_claims
        }
    
    def run_baseline_evaluation(self, sample_size: int = 5) -> Dict:
        """Run baseline evaluation matching Phi4 questions"""
        print(f"\n🎯 GPT-4o Mini Baseline Evaluation ({sample_size} samples)")
        print("=" * 50)
        
        questions = [
            {
                "id": "q1", "question": "What is a Kubernetes Pod and how does it differ from a container?",
                "keywords": {"essential": ["pod", "container", "shared"], "important": ["networking", "storage", "lifecycle"]}
            },
            {
                "id": "q2", "question": "Explain Kubernetes Services and their main types",
                "keywords": {"essential": ["service", "clusterip", "nodeport"], "important": ["loadbalancer", "routing", "endpoint"]}
            },
            {
                "id": "q3", "question": "What is a Kubernetes Deployment and why is it important?",
                "keywords": {"essential": ["deployment", "replicaset", "scaling"], "important": ["rolling update", "rollback", "desired state"]}
            },
            {
                "id": "q4", "question": "What are Kubernetes Namespaces and when should you use them?",
                "keywords": {"essential": ["namespace", "isolation", "virtual"], "important": ["multi-tenancy", "environment", "separation"]}
            },
            {
                "id": "q5", "question": "Explain the difference between ConfigMaps and Secrets in Kubernetes",
                "keywords": {"essential": ["configmap", "secret", "configuration"], "important": ["sensitive", "base64", "environment"]}
            }
        ][:sample_size]
        
        results = []
        
        for i, q in enumerate(questions, 1):
            print(f"\n📝 Question {i}/{len(questions)}: {q['id']}")
            print(f"❓ {q['question']}")
            
            response, response_time = self.query_gpt4o_mini(q['question'])
            
            if response.startswith("ERROR"):
                print(f"❌ {response}")
                continue
            
            scores = self.score_response(response, q['keywords'])
            word_count = len(response.split())
            
            print(f"✅ {word_count} words, {response_time}s")
            print(f"📊 Acc:{scores['accuracy']:.2f} Comp:{scores['completeness']:.2f} Rel:{scores['relevance']:.2f} Tech:{scores['technical_depth']:.2f}")
            
            result = {
                "question_id": q['id'],
                "question": q['question'],
                "response": response,
                "response_time": response_time,
                "word_count": word_count,
                "scores": scores,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
            # Save individual sample
            self.save_sample(f"baseline_{q['id']}", {
                "question": q['question'],
                "response": response,
                "scores": scores,
                "metadata": {
                    "model": self.model,
                    "response_time": response_time,
                    "word_count": word_count,
                    "timestamp": datetime.now().isoformat()
                }
            })
        
        return {"baseline_evaluation": results}
    
    def score_response(self, response: str, keywords: Dict) -> Dict:
        """Score response quality using same method as Phi4 evaluation"""
        response_lower = response.lower()
        
        # Count keywords
        essential_hits = sum(1 for kw in keywords["essential"] if kw.lower() in response_lower)
        important_hits = sum(1 for kw in keywords["important"] if kw.lower() in response_lower)
        
        # Calculate scores
        accuracy = (essential_hits / len(keywords["essential"])) * 0.7 + (important_hits / len(keywords["important"])) * 0.3
        
        word_count = len(response.split())
        completeness = min(1.0, word_count / 100)  # 100 words = full score
        
        k8s_terms = ['kubernetes', 'k8s', 'pod', 'container', 'service', 'deployment']
        relevance = min(1.0, sum(1 for term in k8s_terms if term in response_lower) / 4)
        
        technical_depth = min(1.0, (response_lower.count('because') + response_lower.count('provides') + response_lower.count('enables')) / 3)
        
        return {
            "accuracy": round(min(1.0, accuracy), 3),
            "completeness": round(completeness, 3),
            "relevance": round(relevance, 3),
            "technical_depth": round(technical_depth, 3)
        }
    
    def save_sample(self, sample_id: str, data: Dict):
        """Save individual sample to prevent data loss"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"samples/{sample_id}_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save sample {sample_id}: {e}")
    
    def run_full_evaluation(self, baseline_samples: int = 5) -> Dict:
        """Run complete evaluation: K8s 1.32 detection + baseline"""
        print("🎯 GPT-4o Mini Complete Evaluation")
        print("=" * 60)
        
        # Phase 1: Kubernetes 1.32 knowledge detection
        k8s_132_results = self.check_k8s_132_knowledge()
        
        # Phase 2: Baseline evaluation 
        baseline_results = self.run_baseline_evaluation(baseline_samples)
        
        # Combine results
        all_results = {**k8s_132_results, **baseline_results}
        
        # Generate summary report
        if baseline_results["baseline_evaluation"]:
            baseline_scores = baseline_results["baseline_evaluation"]
            overall_score = statistics.mean([
                statistics.mean([r['scores']['accuracy'] for r in baseline_scores]),
                statistics.mean([r['scores']['completeness'] for r in baseline_scores]),
                statistics.mean([r['scores']['relevance'] for r in baseline_scores]),
                statistics.mean([r['scores']['technical_depth'] for r in baseline_scores])
            ])
            
            report = {
                "evaluation_metadata": {
                    "model": self.model,
                    "evaluation_date": datetime.now().isoformat(),
                    "k8s_132_questions": len(k8s_132_results["k8s_132_detection"]),
                    "baseline_questions": len(baseline_results["baseline_evaluation"]),
                    "total_questions": len(k8s_132_results["k8s_132_detection"]) + len(baseline_results["baseline_evaluation"])
                },
                "k8s_132_analysis": {
                    "detection_results": k8s_132_results["k8s_132_detection"],
                    "overall_confidence": self.assess_overall_k8s_132_confidence(k8s_132_results["k8s_132_detection"])
                },
                "performance_metrics": {
                    "avg_response_time": round(statistics.mean([r['response_time'] for r in baseline_scores]), 2),
                    "avg_word_count": round(statistics.mean([r['word_count'] for r in baseline_scores]), 1)
                },
                "quality_metrics": {
                    "overall_score": round(overall_score, 3),
                    "avg_accuracy": round(statistics.mean([r['scores']['accuracy'] for r in baseline_scores]), 3),
                    "avg_completeness": round(statistics.mean([r['scores']['completeness'] for r in baseline_scores]), 3),
                    "avg_relevance": round(statistics.mean([r['scores']['relevance'] for r in baseline_scores]), 3),
                    "avg_technical_depth": round(statistics.mean([r['scores']['technical_depth'] for r in baseline_scores]), 3)
                },
                "detailed_results": all_results
            }
            
            # Save comprehensive report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f'results/gpt4o_mini_complete_{timestamp}.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            with open('results/gpt4o_mini_latest.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            self.print_summary(report)
            return report
        
        return {}
    
    def assess_overall_k8s_132_confidence(self, k8s_results: List[Dict]) -> str:
        """Assess overall confidence in Kubernetes 1.32 knowledge"""
        high_confidence = sum(1 for r in k8s_results if "HIGH" in r['analysis']['confidence'])
        medium_confidence = sum(1 for r in k8s_results if "MEDIUM" in r['analysis']['confidence'])
        low_confidence = sum(1 for r in k8s_results if "LOW" in r['analysis']['confidence'])
        
        if high_confidence >= len(k8s_results) // 2:
            return "HIGH - Likely trained on K8s 1.32 data"
        elif medium_confidence >= len(k8s_results) // 2:
            return "MEDIUM - Some K8s 1.32 awareness"
        else:
            return "LOW - Limited K8s 1.32 knowledge"
    
    def print_summary(self, report: Dict):
        """Print evaluation summary"""
        print(f"\n🎯 GPT-4O MINI EVALUATION SUMMARY")
        print("=" * 50)
        
        # Kubernetes 1.32 Analysis
        k8s_confidence = report['k8s_132_analysis']['overall_confidence']
        print(f"🔍 K8s 1.32 Knowledge: {k8s_confidence}")
        
        for result in report['k8s_132_analysis']['detection_results']:
            print(f"   • {result['question_id']}: {result['analysis']['confidence']}")
        
        # Baseline Performance
        print(f"\n📊 Baseline Performance:")
        print(f"   • Questions: {report['evaluation_metadata']['baseline_questions']}")
        print(f"   • Overall Score: {report['quality_metrics']['overall_score']:.3f}/1.000")
        print(f"   • Avg Time: {report['performance_metrics']['avg_response_time']}s")
        print(f"   • Avg Words: {report['performance_metrics']['avg_word_count']}")
        
        # Comparison hint
        print(f"\n💡 Compare with Phi4 baseline (0.754) to see performance difference")
        print(f"💾 Results saved: results/gpt4o_mini_latest.json")
        print(f"📁 Individual samples: samples/ directory")

def main():
    print("🎯 Phase 3: GPT-4o Mini Evaluation")
    print("=" * 40)
    print("This will:")
    print("1. Check if GPT-4o mini has Kubernetes 1.32 knowledge")
    print("2. Run baseline evaluation matching Phi4 tests")
    print("3. Save all samples with caching for resumption")
    print("")
    
    # Get API key
    api_key = input("🔑 Enter your OpenAI API key (or press Enter if set in OPENAI_API_KEY): ").strip()
    if not api_key:
        api_key = None
    
    evaluator = GPT4oMiniEvaluator(api_key=api_key)
    report = evaluator.run_full_evaluation(baseline_samples=5)
    
    if report:
        print("\n🎉 Evaluation complete!")
        print("👉 Ready to compare with Phi4 and plan distillation")
        return 0
    else:
        print("\n❌ Evaluation failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
