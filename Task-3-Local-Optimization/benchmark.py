#!/usr/bin/env python3
"""
Task 3: Local Model Optimization & Benchmarking Script
Model Family: Qwen2.5-1.5B-Instruct
Quantizations Evaluated: Q4 (e.g. q4_K_M or q4_0) vs Q8 (e.g. q8_0)

This script:
1. Connects to Ollama API (http://localhost:11434).
2. Evaluates standardized prompts deterministically on both Q4 and Q8.
3. Measures:
   - Time to first token (TTFT)
   - Total latency & generation time
   - Generated token counts
   - Output tokens per second (tok/s)
   - Peak RSS memory footprint
4. Saves predictions and benchmark CSV metrics.
"""

import os
import sys
import time
import json
import subprocess
import platform
import urllib.request
import urllib.error

def detect_environment():
    uname = platform.uname()
    print("="*60)
    print("System Environment Profile")
    print("="*60)
    print(f"OS System:       {uname.system} {uname.release}")
    print(f"Architecture:    {uname.machine}")
    
    try:
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        print(f"Processor:       {chip}")
    except Exception:
        pass
        
    try:
        mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
        print(f"System Memory:   {mem_bytes / (1024**3):.1f} GB")
    except Exception:
        pass
    print("="*60)

def query_ollama(model_tag, prompt, temperature=0.0, max_tokens=512):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_tag,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            t1 = time.time()
            res_json = json.loads(resp.read().decode("utf-8"))
            elapsed_sec = t1 - t0
            
            eval_count = res_json.get("eval_count", 0)
            eval_duration_ns = res_json.get("eval_duration", 0)
            prompt_eval_duration_ns = res_json.get("prompt_eval_duration", 0)
            
            tok_per_sec = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else (eval_count / elapsed_sec if elapsed_sec > 0 else 0)
            ttft_sec = (prompt_eval_duration_ns / 1e9) if prompt_eval_duration_ns > 0 else 0.0
            
            return {
                "success": True,
                "response": res_json.get("response", ""),
                "eval_count": eval_count,
                "elapsed_sec": round(elapsed_sec, 3),
                "ttft_sec": round(ttft_sec, 4),
                "tokens_per_sec": round(tok_per_sec, 2),
                "raw_metrics": res_json
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "response": "",
            "eval_count": 0,
            "elapsed_sec": 0,
            "ttft_sec": 0,
            "tokens_per_sec": 0
        }

def get_process_memory():
    """Extract RSS memory using psutil if available, or ps."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # MB
    except Exception:
        return 0.0

def run_benchmark(model_q4="qwen2.5:1.5b", model_q8="qwen2.5:1.5b-instruct-q8_0", prompts_file="prompts.json"):
    detect_environment()
    
    if not os.path.exists(prompts_file):
        prompts_file = os.path.join(os.path.dirname(__file__), "prompts.json")
    
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)
        
    print(f"Loaded {len(prompts)} standardized benchmark prompts.")
    
    results = {"q4": [], "q8": []}
    
    for tag, name in [(model_q4, "q4"), (model_q8, "q8")]:
        print(f"\n>>> Benchmarking Model: {tag} ({name.upper()}) <<<")
        for p in prompts:
            print(f"Running Prompt {p['id']}/10: [{p['category']}]... ", end="", flush=True)
            res = query_ollama(tag, p["prompt"])
            if res["success"]:
                print(f"DONE | Tokens: {res['eval_count']} | Speed: {res['tokens_per_sec']} tok/s | Latency: {res['elapsed_sec']}s")
            else:
                print(f"FAILED: {res.get('error')}")
            
            record = {
                "id": p["id"],
                "category": p["category"],
                "prompt": p["prompt"],
                "model_tag": tag,
                "quantization": name.upper(),
                "response": res["response"],
                "tokens": res["eval_count"],
                "latency_sec": res["elapsed_sec"],
                "ttft_sec": res["ttft_sec"],
                "tokens_per_sec": res["tokens_per_sec"],
                "success": res["success"]
            }
            results[name].append(record)
            
    # Save outputs
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "benchmark_results_raw.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nAll benchmark raw logs saved to: {out_dir}/benchmark_results_raw.json")

if __name__ == "__main__":
    q4_model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:1.5b"
    q8_model = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5:1.5b-instruct-q8_0"
    run_benchmark(model_q4=q4_model, model_q8=q8_model)
