# LLM Orchestration Framework

<p align="center">
  <strong>General-purpose LLM orchestration layer — define agents, chain them together, route tasks, handle escalation</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#deployment">Deployment</a> •
  <a href="#license">License</a>
</p>

## 🚀 Overview

The **LLM Orchestration Framework** is a production-ready Python library for building intelligent agent systems. It provides a flexible, provider-agnostic orchestration layer that lets you define specialized agents, chain them together, route tasks based on capabilities, and handle escalations gracefully.

### Why Use This Framework?

- **🧩 Agent-Centric Design**: Define agents with specific capabilities and let the framework route tasks intelligently
- **🔗 Powerful Chaining**: Sequential, parallel, conditional, and fallback chain execution
- **🎯 Smart Routing**: Capability-based task routing with confidence scoring
- **🧠 Built-in Memory**: Conversation buffers and automatic summarization
- **⚡ Production Ready**: Async-first, type-hinted, fully tested, and extensible
- **🌐 Multi-Provider**: Support for OpenAI, Anthropic, DeepSeek, Ollama, and custom providers

## ✨ Features

- **Agent Registry**: Register, discover, and manage agents with specific capabilities
- **Task Router**: Intelligent capability matching with exact→partial→fuzzy matching
- **Chain Engine**: Sequential, parallel, conditional, and fallback chain execution
- **Orchestrator**: Central management with task lifecycle, stats, and escalation
- **Memory Systems**: Conversation buffers and automatic summarization
- **REST API**: Full-featured FastAPI server with OpenAPI documentation
- **Multi-Provider**: OpenAI, Anthropic, DeepSeek, Ollama support + easy extension
- **Async-First**: Built on asyncio for high-concurrency workloads
- **Type-Safe**: Full Python type hints with Pydantic v2 validation

## 🏁 Quick Start

### Installation

```bash
# Using pip
pip install llm-orchestration-framework

# Using uv (recommended)
uv add llm-orchestration-framework
```

### Basic Usage

```python
import asyncio
from llm_orchestration_framework import Orchestrator, AgentConfig, Task
from llm_orchestration_framework.providers import OpenAIProvider

async def main():
    # Create orchestrator
    orchestrator = Orchestrator()
    await orchestrator.start()
    
    # Create a provider
    provider = OpenAIProvider(api_key="your-api-key")
    
    # Create and register an agent
    config = AgentConfig(
        id="classifier",
        name="Text Classifier",
        capabilities=["classification", "sentiment_analysis"],
        provider="openai",
        model="gpt-4o-mini",
        system_prompt="You are a text classification expert."
    )
    
    # Submit a task
    task = Task(
        input="Classify this sentiment: I love this product!",
        metadata={"capabilities": ["classification"]}
    )
    
    task_id = await orchestrator.submit_task(task)
    
    # Get result
    result = await orchestrator.get_result(task_id)
    print(f"Result: {result.output}")
    print(f"Confidence: {result.confidence}")
    
    await orchestrator.stop()

asyncio.run(main())
```

### Running the API Server

```bash
# Start the server
uvicorn llm_orchestration_framework.api.server:app --host 0.0.0.0 --port 8000 --reload

# Or with docker
docker-compose up
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Tasks   │  │ Registry │  │  Router  │  │  Chain   │   │
│  │          │  │          │  │          │  │  Engine  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Agent Layer                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Agent A │  │  Agent B │  │  Agent C │  │  Agent D │   │
│  │(Classifier)│(Code Gen) │ (Summarizer)│ (Translator)│   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Provider Layer                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  OpenAI  │  │ Anthropic│  │ DeepSeek │  │  Ollama  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Task**: Unit of work with input, context, metadata, and configuration
2. **Agent**: Specialized worker with capabilities, provider, and configuration
3. **Registry**: Central directory for discovering and managing agents
4. **Router**: Matches tasks to agents based on capabilities and health
5. **Chain Engine**: Executes sequences of tasks with various patterns
6. **Orchestrator**: Manages lifecycle, statistics, and escalations
7. **Memory**: Conversation history buffers and automatic summarization
8. **Providers**: LLM API integrations (OpenAI, Anthropic, DeepSeek, Ollama)

## 📚 API Reference

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /v1/tasks` | POST | Submit a task for execution |
| `GET /v1/tasks/{task_id}` | GET | Get task result |
| `POST /v1/agents/register` | POST | Register a new agent |
| `GET /v1/agents` | GET | List all agents |
| `GET /v1/agents/{agent_id}` | GET | Get agent details |
| `POST /v1/chains/sequential` | POST | Run sequential chain |
| `POST /v1/chains/parallel` | POST | Run parallel chain |
| `GET /v1/stats` | GET | Get orchestrator statistics |
| `GET /health` | GET | Health check |
| `GET /.well-known/ai-agent.json` | GET | AI agent manifest |

### Python API

#### Core Classes

```python
# Task management
from llm_orchestration_framework import Task, TaskResult, TaskStatus

# Agent system  
from llm_orchestration_framework import AgentConfig, BaseAgent, AgentRegistry

# Orchestration
from llm_orchestration_framework import Orchestrator, TaskRouter, ChainEngine

# Memory
from llm_orchestration_framework import ConversationBuffer, ConversationSummary

# Providers
from llm_orchestration_framework.providers import (
    OpenAIProvider, AnthropicProvider, 
    DeepSeekProvider, OllamaProvider
)
```

#### Example: Creating a Custom Agent

```python
from llm_orchestration_framework import BaseAgent, AgentConfig
from llm_orchestration_framework.providers import OpenAIProvider

class TranslationAgent(BaseAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.provider = OpenAIProvider()
    
    async def _execute_internal(self, task):
        # Custom execution logic
        prompt = f"Translate to French: {task.input}"
        response = await self.provider.generate(prompt, model="gpt-4")
        
        return TaskResult(
            task_id=task.id,
            agent_id=self.config.id,
            output=response,
            confidence=0.95,
            tokens_used=self.provider.count_tokens(response),
            processing_time_ms=0,  # Will be filled by base class
            status=TaskStatus.COMPLETED
        )
```

## 🐳 Deployment

### Docker

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY . .

CMD ["uvicorn", "llm_orchestration_framework.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - OLLAMA_HOST=http://ollama:11434
    volumes:
      - ./data:/app/data
  
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  ollama_data:
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required |
| `DEEPSEEK_API_KEY` | DeepSeek API key | Required |
| `OLLAMA_HOST` | Ollama endpoint | `http://localhost:11434` |
| `LOG_LEVEL` | Logging level | `INFO` |

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=llm_orchestration_framework --cov-report=html
```

## 🔧 Development

### Project Structure

```
llm-orchestration-framework/
├── core/                    # Core orchestration logic
│   ├── task.py             # Task models
│   ├── agent.py            # Agent models and base
│   ├── registry.py         # Agent registry
│   ├── router.py           # Task router
│   ├── chain.py            # Chain engine
│   └── orchestrator.py     # Main orchestrator
├── providers/              # LLM providers
│   ├── base.py             # Base provider class
│   ├── openai.py           # OpenAI provider
│   ├── anthropic.py        # Anthropic provider
│   ├── deepseek.py         # DeepSeek provider
│   └── ollama.py           # Ollama provider
├── memory/                 # Memory systems
│   ├── base.py             # Base memory class
│   ├── buffer.py           # Conversation buffer
│   └── summary.py          # Auto-summarization
├── api/                    # REST API
│   ├── server.py           # FastAPI app
│   └── schemas.py          # Pydantic schemas
├── tests/                  # Test suite
│   ├── conftest.py         # Test fixtures
│   └── test_framework.py   # Comprehensive tests
├── pyproject.toml          # Project configuration
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker compose
├── .env.example            # Environment template
└── README.md               # This file
```

### Adding a New Provider

1. Create a new file in `providers/`
2. Extend `BaseProvider` class
3. Implement `generate()` and `count_tokens()` methods
4. Add to `providers/__init__.py`
5. Write tests

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit a PR

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/AnaVuko1/llm-orchestration-framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AnaVuko1/llm-orchestration-framework/discussions)
- **Email**: ana@example.com

---

<p align="center">
  Built with ❤️ by Ana Vukojevic
</p>