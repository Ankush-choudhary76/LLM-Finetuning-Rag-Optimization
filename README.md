# LLM Engineering Assignment — Complete Repository

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.44+-yellow.svg)](https://github.com/huggingface/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-green.svg)](https://github.com/huggingface/peft)
[![Zero Framework RAG](https://img.shields.io/badge/RAG-Zero%20Framework-red.svg)](https://faiss.ai/)
[![Local Inference](https://img.shields.io/badge/Inference-Local%20Ollama-purple.svg)](https://ollama.com/)

A production-grade, technically rigorous implementation of three core Large Language Model (LLM) engineering capabilities:
1. **Task 1: LoRA / QLoRA Parameter-Efficient Fine-Tuning** (Natural Language to Structured JSON extraction on `Qwen2.5-0.5B-Instruct` + 7B parameter VRAM mathematical derivation).
2. **Task 2: RAG Pipeline Engineered from Scratch** (Built over the 48-page *NIST AI 100-1* standard without LangChain/LlamaIndex, featuring FAISS dense search, BM25 keyword matching, Reciprocal Rank Fusion, Cross-Encoder reranking, and grounded generation with source citations).
3. **Task 3: Local Model Optimization & Benchmarking** (Quantization comparison between Q4 and Q8 on `Qwen2.5-1.5B-Instruct`, memory profiling, KV cache deep dive, and a production 20-concurrent-user serving architecture on a 24GB GPU).

---

## Repository Structure

```
LLM-Engineering-Assignment/
├── Task-1-Fine-Tuning/
│   ├── Task_1_LoRA_Fine_Tuning.ipynb    # Complete runnable Colab notebook
│   ├── datasets/
│   │   ├── generate_dataset.py          # Deterministic data generation script (seed=42)
│   │   ├── train.json                   # 600 training samples
│   │   ├── val.json                     # 150 validation samples
│   │   └── test.json                    # 150 held-out test samples
│   ├── adapters/                        # Target directory for saved PEFT LoRA adapters
│   └── results/                         # Evaluation tables, failure analyses, 7B memory chart
│
├── Task-2-RAG/
│   ├── Task_2_RAG_From_Scratch.ipynb    # Framework-free Colab notebook
│   ├── eval_questions.json              # 15 verified ground-truth questions with source pages
│   ├── documents/                       # Ingested PDF directory (NIST.AI.100-1.pdf, 48 pages)
│   ├── indexes/                         # Serialized FAISS indices (300t & 500t configs)
│   └── results/                         # Retrieval comparison metrics (Recall@1, Recall@5, MRR)
│
├── Task-3-Local-Optimization/
│   ├── Task_3_Local_Model_Optimization.ipynb # Local evaluation notebook
│   ├── benchmark.py                     # Python benchmarking script for macOS M2
│   ├── prompts.json                     # 10 standardized reasoning and coding benchmark prompts
│   └── results/                         # Benchmark latency, throughput, and memory logs
│
├── reports/
│   └── LLM_Engineering_Assignment_Report.md # Comprehensive final technical report
│
├── requirements.txt                     # Pinned project dependencies
└── README.md                            # Comprehensive execution guide and overview
```

---

## Quickstart & Execution Instructions

### Environment Setup

#### Option A: Local Setup (macOS M2 / Linux)
```bash
# Clone and enter directory
cd "LLM-Engineering-Assignment"

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Option B: Google Colab Setup (Tasks 1 & 2)
1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Upload `Task_1_LoRA_Fine_Tuning.ipynb` or `Task_2_RAG_From_Scratch.ipynb`.
4. Run cells sequentially from top to bottom. Each notebook contains self-contained dependency installations, dataset generation, validation checks, and automatic ZIP export of results.

---

### Task 1: LoRA Fine-Tuning Execution
- **Notebook:** `Task-1-Fine-Tuning/Task_1_LoRA_Fine_Tuning.ipynb`
- **Objective:** Fine-tune `Qwen/Qwen2.5-0.5B-Instruct` on Natural Language $\to$ Structured JSON slot filling.
- **Workflow:**
  1. Inspects model architecture linear projections (`q_proj`, `v_proj`).
  2. Generates deterministic dataset (600 train, 150 val, 150 test).
  3. Trains LoRA adapter ($r=8, \alpha=16$, lr=$2\times 10^{-4}$, effective batch size 16, 3 epochs).
  4. Evaluates base vs. fine-tuned model on identical held-out test data.
  5. Computes JSON validity %, exact match accuracy, and field-level accuracy.
  6. Automatically identifies and diagnoses authentic failure cases.
  7. Derives complete theoretical VRAM comparison for a 7B model.

---

### Task 2: Framework-Free RAG Execution
- **Notebook:** `Task-2-RAG/Task_2_RAG_From_Scratch.ipynb`
- **Target Document:** *NIST AI 100-1: Artificial Intelligence Risk Management Framework (AI RMF 1.0)* (48 pages).
- **Workflow:**
  1. Downloads and validates the 48-page NIST publication.
  2. Extracts text page-by-page, preserving metadata.
  3. Compares two chunking configurations: **300 tokens** vs. **500 tokens** (both with 50-token overlap).
  4. Indexes vectors into FAISS (`all-MiniLM-L6-v2`) and builds BM25 keyword indices.
  5. Performs hybrid retrieval with Reciprocal Rank Fusion (RRF, $k=60$).
  6. Reranks candidate passages using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
  7. Evaluates Recall@1, Recall@5, MRR, and latency on 15 verified benchmark questions.
  8. Generates grounded answers with explicit `[Page X]` source citations using `Qwen2.5-0.5B-Instruct`.

---

### Task 3: Local Model Benchmarking
- **Script:** `Task-3-Local-Optimization/benchmark.py`
- **Notebook:** `Task-3-Local-Optimization/Task_3_Local_Model_Optimization.ipynb`
- **Model Family:** `Qwen2.5-1.5B-Instruct` (Q4_K_M vs. Q8_0).

#### Steps to Execute:
```bash
# 1. Start Ollama in background
ollama serve

# 2. Pull Q4 and Q8 quantized models
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:1.5b-instruct-q8_0

# 3. Run the automated benchmark script
python3 Task-3-Local-Optimization/benchmark.py "qwen2.5:1.5b" "qwen2.5:1.5b-instruct-q8_0"
```

The script evaluates 10 standardized prompts covering coding, mathematics, logic, and extraction, recording TTFT, tokens-per-second, and memory footprint into `Task-3-Local-Optimization/results/`.

---

## Key Experimental Results Summary

### Task 1: Fine-Tuning Performance (150 Held-Out Test Samples, Actual Run)
- **JSON Validity Rate:** Base: **100.00%** | Fine-Tuned: **90.67%**
- **Field-Level Slot Accuracy:** Base: **41.81%** $\to$ Fine-Tuned: **58.89%** ($+17.08\%$ improvement)

### Task 2: Retrieval Evaluation (15 Ground-Truth NIST Queries, Actual Run)
- **300-Token Chunks:** Recall@5: **6.67%** | MRR: **0.0222** | Latency: 71.13 ms
- **500-Token Chunks:** Recall@5: **13.33%** | MRR: **0.0444** | Latency: 86.26 ms

### Task 3: Local Model Optimization Benchmarks (Measured Results)
- **Model Disk Size:** Q4: **986 MB** vs. Q8: **1.60 GB** ($+66.7\%$)
- **Peak Runtime RSS:** Q4: **~1.35 GB** vs. Q8: **~2.28 GB** ($+68.8\%$)
- **Generation Speed:** Q4: **66.82 tok/s** vs. Q8: **44.04 tok/s** ($+51.7\%$ faster throughput for Q4)
- **Measured Range:** Q4: $64.44 - 68.77$ tok/s | Q8: $42.93 - 45.22$ tok/s
- **Output Quality:** Q8 preserved cleaner step-by-step mathematical derivations; both achieved 100% schema accuracy on structured extraction.

---

## Submission & Interview Checklist

- [x] **Task 1 Notebook:** Top-to-bottom executable, installs dependencies, handles GPU/CPU, generates deterministic datasets, trains LoRA adapter, evaluates on held-out test data, includes real failure analysis, and contains complete 7B VRAM derivation.
- [x] **Task 2 Notebook:** Strict zero-framework architecture (no LangChain/LlamaIndex), page-by-page PDF extraction with metadata tracking, 300t vs 500t chunk experiment, FAISS + BM25 hybrid search with RRF, Cross-Encoder reranker, 15 verified ground-truth queries, Recall@1/Recall@5/MRR metrics, retrieval failure analysis, and conceptual writeups.
- [x] **Task 3 Notebook & Benchmark:** Local environment detection, Q4 vs Q8 quantization benchmarking on `Qwen2.5-1.5B-Instruct`, memory profiling, 10 standardized reasoning prompts, KV cache deep dive, and a production 20-user serving architecture on a 24GB GPU.
- [x] **Comprehensive Report:** `reports/LLM_Engineering_Assignment_Report.md` complete with executive summary, methodology, mathematical derivations, empirical tables, and serving architecture.
- [x] **Requirements & Modular Code:** `requirements.txt` and clean directory structure ready for evaluation.

