# LLM Notes

A collection of notes, examples, and experiments with Large Language Models (LLMs), covering theoretical concepts, practical applications, and implementation details.

## Table of Contents

### Core Concepts
| Topic | Description | Link |
|-------|-------------|------|
| AI Engineering vs ML Engineering | Comparison of AI and ML engineering approaches | [Link](./AI%20Engineering%20Versus%20ML%20Engineering.md) |
| Agents | Implementation and concepts of LLM agents | [Link](./Agents.md) |
| More than just a next token predictor | Deeper exploration of LLM capabilities | [Link](./More%20than%20just%20a%20next%20token%20predictor.md) |
| Productionizing an ML system | Guide for deploying ML systems to production | [Link](./Productionizing%20an%20ML%20system.md) |

### Practical Applications
| Topic | Description | Link |
|-------|-------------|------|
| Function Calling | Implementing function calling with LLMs | [Link](./Function%20Calling.ipynb) |
| Pydantic | Using Pydantic for data validation with LLMs | [Link](./Pydantic.ipynb) |
| Pydantic Bank Support | Example of Pydantic for financial services | [Link](./Pydantic_Bank_support.ipynb) |
| Structured Extraction | Extracting structured data with LLMs | [Link](./Structured%20Extraction.ipynb) |
| Structured Extraction with Llama | Using Llama for data extraction | [Link](./Structured%20Extraction%20with%20llama.ipynb) |
| Record Linkage | Entity resolution and record matching with LLMs | [Link](./Record%20Linkage.ipynb) |
| Llama Index | Working with the LlamaIndex library | [Link](./llama%20Index.ipynb) |
| Ollama | Using Ollama for local LLM deployment | [Link](./ollama.ipynb) |

### Theory
| Topic | Description | Link |
|-------|-------------|------|
| Attention | Deep dive into attention mechanisms | [Link](./Theory/Attention.ipynb) |
| Reinventing Attention | First principles approach to attention | [Link](./Theory/Reinventing%20Attention%20from%20First%20Principles.ipynb) |
| Prompting | Techniques for effective prompting | [Link](./Theory/Prompting.ipynb) |
| Prompting LLMs | Advanced prompting strategies | [Link](./Theory/Prompting%20LLMs.ipynb) |
| Sampling | Sampling methods for text generation | [Link](./Theory/Sampling.ipynb) |
| Evaluation | Metrics and methods for LLM evaluation | [Link](./Theory/Evaluation.ipynb) |
| LLM Classification | Using LLMs for classification tasks | [Link](./Theory/LLM%20Classification.ipynb) |

### Reinforcement Learning
| Topic | Description | Link |
|-------|-------------|------|
| Introduction | Basics of reinforcement learning | [Link](./RL/0.ipynb) |
| Fundamentals | Core RL concepts and algorithms | [Link](./RL/1.ipynb) |
| Advanced Topics | More complex RL techniques | [Link](./RL/2.ipynb) |
| Applications | Applying RL to practical problems | [Link](./RL/3.ipynb) |

## Work in Progress
Some notebooks are still under development. Feel free to explore, but note that certain content may be incomplete.

## Setup
This project uses a Python virtual environment. To set it up:

```bash
# Clone the repository
git clone <repository-url>

# Navigate to the project directory
cd LLM-notes

# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt  # If available
```