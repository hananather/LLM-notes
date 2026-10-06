# LLM Notes

Curated notes, notebooks, and small demos covering LLM concepts, patterns, and agent architectures. Designed for quick learning and reproducible experiments.

## Start Here
- Requirements: Python 3.11 (see `.python-version`).
- Setup (recommended):
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install jupyter
  jupyter lab  # or: jupyter notebook
  ```
- Execute a notebook headlessly (smoke test):
  ```bash
  jupyter nbconvert --to notebook --execute path/to/notebook.ipynb --inplace
  ```

## Table of Contents

### Core Concepts
| Topic | Description | Link |
|-------|-------------|------|
| AI Engineering vs ML Engineering | Comparison of AI and ML engineering approaches | [Link](./Systems/ai-engineering-versus-ml-engineering.md) |
| Agents | Implementation and concepts of LLM agents | [Link](./Agents/README.md) |
| More than just a next token predictor | Deeper exploration of LLM capabilities | [Link](./Theory/prompting-llms.ipynb) |
| Productionizing an ML system | Guide for deploying ML systems to production | [Link](./Systems/productionizing-an-ml-system.md) |

### Practical Applications
| Topic | Description | Link |
|-------|-------------|------|
| Function Calling | Implementing function calling with LLMs | [Link](./LLM_basics/function-calling.ipynb) |
| Structured Output | Emitting JSON and schemas from LLMs | [Link](./LLM_basics/structured-output.ipynb) |
| Structured Extraction | Extracting structured data with LLMs | [Link](./LLM_basics/structured-extraction.ipynb) |
| Structured Extraction with Llama | Using Llama for data extraction | [Link](./LLM_basics/structured-extraction-with-llama.ipynb) |
| Record Linkage | Entity resolution and record matching with LLMs | [Link](./LLM_basics/record-linkage.ipynb) |
| Record-linkage handbook | Splink–LOTUS companion with FEBRL, BioDEX and Wikipedia experiments; eight chapters and nine TikZ figures | [Notebook](./LLM_basics/record-linkage-with-semantic-operators.ipynb) |
| Pydantic | Using Pydantic for data validation with LLMs | [Link](./LLM_basics/pydantic.ipynb) |

### Model Context Protocol
| Topic | Description | Link |
|-------|-------------|------|
| MCP notes | Servers, clients, reference servers, resources, and prompts | [Link](./MCP/README.md) |
| Direct tool calling | Call arXiv tools in the notebook process | [Link](./MCP/mcp.ipynb) |
| MCP server | Expose those tools over stdio | [Link](./MCP/mcp-server.ipynb) |
| MCP client | Connect one chatbot to the research server | [Link](./MCP/mcp-client.ipynb) |
| Reference servers | Connect one chatbot to several MCP servers | [Link](./MCP/reference-servers.ipynb) |
| Prompts and resources | Read-only paper data and a search prompt | [Link](./MCP/prompts-and-resources.ipynb) |

### More folders
| Topic | Description | Link |
|-------|-------------|------|
| LangGraph | Small graph and agent experiments | [Link](./LangGraph/README.md) |
| DSPy | DSPy notes and drafts | [Link](./DSPy/README.md) |
| Prediction-powered inference | Estimator notebooks | [Link](./PPI/README.md) |

### Theory
| Topic | Description | Link |
|-------|-------------|------|
| Attention | Deep dive into attention mechanisms | [Link](./Theory/attention.ipynb) |
| Reinventing Attention | First principles approach to attention | [Link](./Theory/reinventing-attention-from-first-principles.ipynb) |
| Prompting | Techniques for effective prompting | [Link](./Theory/prompting.ipynb) |
| Prompting LLMs | Advanced prompting strategies | [Link](./Theory/prompting-llms.ipynb) |
| Sampling | Sampling methods for text generation | [Link](./Theory/sampling.ipynb) |
| Evaluation | Metrics and methods for LLM evaluation | [Link](./Theory/evaluation.ipynb) |
| LLM Classification | Using LLMs for classification tasks | [Link](./Theory/llm-classification.ipynb) |

### Reinforcement Learning
| Topic | Description | Link |
|-------|-------------|------|
| Introduction | Basics of reinforcement learning | [Link](./RL/0.ipynb) |
| Fundamentals | Core RL concepts and algorithms | [Link](./RL/1.ipynb) |
| Advanced Topics | More complex RL techniques | [Link](./RL/2.ipynb) |
| Applications | Applying RL to practical problems | [Link](./RL/3.ipynb) |
| Policy Gradients | Implementing and analyzing policy gradients | [Link](./RL/4.ipynb) |
| Additional Topics | Extensions and case studies | [Link](./RL/5.ipynb) |

## Contributing & Naming
- Read the contributor guide: [`agents.md`](./agents.md).
- New docs, notebooks, and images should use kebab-case (e.g., `structured-extraction.ipynb`).
- Python modules should use snake_case to remain importable (e.g., `research_server.py`).

## Notes
- Some notebooks are work-in-progress; run cells top-to-bottom and install missing libraries as prompted.
- Keep outputs lean to improve diffs and review.
