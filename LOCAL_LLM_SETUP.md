# Local LLM Setup

Local-only RAG pipeline using Ollama models.

## Setup

```bash
# Install Ollama
brew install ollama

# Start service
ollama serve

# Download model
ollama pull llama3.2:3b
```

## Usage

```python
from src.generation import RAGGenerator

generator = RAGGenerator()  # Uses llama3.2:3b
result = generator.generate(query, retrieved_docs)
```

## Models

| Model | RAM | Quality |
|-------|-----|---------|
| `llama3.2:3b` | 2GB | Good |
| `llama3.1:8b` | 5GB | Better |
| `mistral:7b` | 4GB | Better |

## Troubleshooting

- Model not found: `ollama pull <model>`
- Connection refused: `ollama serve`
- Out of memory: Use smaller model