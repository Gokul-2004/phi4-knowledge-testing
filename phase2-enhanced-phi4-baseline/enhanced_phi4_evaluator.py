#!/usr/bin/env python3
import subprocess
import time
import json
import sys
from datetime import datetime
from pathlib import Path
import statistics

class EnhancedPhi4Evaluator:
    def __init__(self):
        Path("results").mkdir(exist_ok=True)
        Path("cache").mkdir(exist_ok=True)
        
    def check_ollama_phi4(self):
        print("🔍 Checking Ollama and Phi4...")
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print("❌ Ollama not found. Install: curl -fsSL https://ollama.ai/install.sh | sh")
                return False
            
            if 'phi4' not in result.stdout.lower():
                print("💡 Installing Phi4...")
                subprocess.run(['ollama', 'pull', 'phi4:latest'], timeout=600)
            
            print("✅ Ready!")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def query_phi4(self, question):
        print("🤖 Querying Phi4...")
        start_time = time.time()
        try:
            result = subprocess.run(['ollama', 'run', 'phi4:latest', question], 
                                  capture_output=True, text=True, timeout=300)
            response_time = round(time.time() - start_time, 2)
            
            if result.returncode == 0:
                return result.stdout.strip(), response_time
            else:
                return f"ERROR: {result.stderr}", response_time
        except Exception as e:
            return f"ERROR: {e}", round(time.time() - start_time, 2)
    
    def score_response(self, response, keywords):
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
    
    def run_evaluation(self, sample_size=5):
        print("🎯 Enhanced Phi4 Baseline Evaluation")
        print("=" * 50)
        
        if not self.check_ollama_phi4():
            return {}
        
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
            
            response, response_time = self.query_phi4(q['question'])
            
            if response.startswith("ERROR"):
                print(f"❌ {response}")
                continue
            
            scores = self.score_response(response, q['keywords'])
            word_count = len(response.split())
            
            print(f"✅ {word_count} words, {response_time}s")
            print(f"📊 Acc:{scores['accuracy']:.2f} Comp:{scores['completeness']:.2f} Rel:{scores['relevance']:.2f} Tech:{scores['technical_depth']:.2f}")
            
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
            overall_score = statistics.mean([
                statistics.mean([r['scores']['accuracy'] for r in results]),
                statistics.mean([r['scores']['completeness'] for r in results]),
                statistics.mean([r['scores']['relevance'] for r in results]),
                statistics.mean([r['scores']['technical_depth'] for r in results])
            ])
            
            report = {
                "evaluation_metadata": {
                    "model": "phi4:latest",
                    "evaluation_date": datetime.now().isoformat(),
                    "total_questions": len(results),
                    "success_rate": 1.0
                },
                "performance_metrics": {
                    "avg_response_time": round(statistics.mean([r['response_time'] for r in results]), 2),
                    "avg_word_count": round(statistics.mean([r['word_count'] for r in results]), 1)
                },
                "quality_metrics": {
                    "overall_score": round(overall_score, 3),
                    "avg_accuracy": round(statistics.mean([r['scores']['accuracy'] for r in results]), 3),
                    "avg_completeness": round(statistics.mean([r['scores']['completeness'] for r in results]), 3),
                    "avg_relevance": round(statistics.mean([r['scores']['relevance'] for r in results]), 3),
                    "avg_technical_depth": round(statistics.mean([r['scores']['technical_depth'] for r in results]), 3)
                },
                "detailed_results": results
            }
            
            # Save report
            with open('results/phi4_baseline_latest.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n🎯 BASELINE SUMMARY")
            print("=" * 30)
            print(f"📊 Questions: {len(results)}")
            print(f"🎯 Overall Score: {overall_score:.3f}/1.000")
            print(f"⚡ Avg Time: {report['performance_metrics']['avg_response_time']}s")
            print(f"📝 Avg Words: {report['performance_metrics']['avg_word_count']}")
            print(f"💾 Results saved: results/phi4_baseline_latest.json")
            
            return report
        
        return {}

def main():
    evaluator = EnhancedPhi4Evaluator()
    report = evaluator.run_evaluation(sample_size=5)
    
    if report:
        print("\n🎉 Baseline evaluation complete!")
        print("👉 Ready for Phase 3 distillation comparison")
        return 0
    else:
        print("❌ Evaluation failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
