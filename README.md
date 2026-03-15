# Agentic LLM Chip Design Assistant

[![CI](https://github.com/YashShevkar30/agentic-chip-design-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/YashShevkar30/agentic-chip-design-assistant/actions)

An AI agent capable of interpreting Hardware Description Languages (HDL) and suggesting
optimizations for chip design modules using RAG and speculative decoding.

## 🎯 Problem Statement
Chip designers spend hours manually reviewing RTL for optimization opportunities.
This agent analyzes Verilog/VHDL, detects issues, and suggests improvements
using retrieval-augmented generation over 50k+ engineering papers.

## 🏗️ Architecture
```
HDL Code Input → Module Parser → Issue Detection → RAG Retrieval → Reranked Suggestions
                                      ↓                  ↓
                               Syntax Checker     FAISS + BM25 Hybrid Search
                               Port Analyzer      Cross-Encoder Reranking
                               Timing Estimator   Speculative Decoding (1.8x)
```

## 🔧 Tech Stack
| Component | Technology |
|-----------|-----------|
| **Agent** | Custom ReAct-style HDL interpreter |
| **RAG** | FAISS + BM25 hybrid search |
| **Reranking** | Cross-encoder (ms-marco-MiniLM) |
| **Scaling** | Speculative decoding (1.8x speedup) |
| **Embeddings** | sentence-transformers |
| **Framework** | LangChain, PyTorch |

## 🔍 Agent Capabilities
- **Module Parsing**: Extract modules, ports, parameters from Verilog
- **Issue Detection**: Blocking assignments, clock domains, incomplete sensitivity
- **Syntax Checking**: Module/endmodule matching, semicolons, begin/end
- **Timing Estimation**: Critical path depth analysis
- **Port Analysis**: Signal width and direction analysis

## 🚀 Quick Start
```bash
pip install -r requirements.txt
pytest tests/ -v
```

## 📄 License
MIT License - Yash Shevkar
