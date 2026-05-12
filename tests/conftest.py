"""
Test fixtures for LLM orchestration framework.
"""

import asyncio
import uuid
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from llm_orchestration_framework.api.server import app
from llm_orchestration_framework.core.agent import AgentConfig, BaseAgent
from llm_orchestration_framework.core.orchestrator import Orchestrator
from llm_orchestration_framework.core.registry import AgentRegistry
from llm_orchestration_framework.core.task import Task, TaskResult, TaskStatus
from llm_orchestration_framework.providers.base import BaseProvider


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    provider = MagicMock(spec=BaseProvider)
    provider.generate = AsyncMock(return_value="Mock response")
    provider.count_tokens = MagicMock(return_value=100)
    return provider


@pytest.fixture
def test_agent_a(mock_provider):
    """Create test agent A with specific capabilities."""
    
    class MockAgent(BaseAgent):
        async def _execute_internal(self, task):
            return TaskResult(
                task_id=task.id,
                agent_id=self.config.id,
                output=f"Agent A response to: {task.input[:30]}",
                confidence=0.9,
                tokens_used=100,
                processing_time_ms=500,
                status=TaskStatus.COMPLETED,
                metadata={"agent": "A", "mock": True}
            )
    
    config = AgentConfig(
        id="agent_a",
        name="Test Agent A",
        description="Test agent for classification",
        capabilities=["classification", "sentiment_analysis"],
        provider="mock",
        model="mock-model",
        system_prompt="You are a test agent.",
        max_concurrency=2
    )
    
    return MockAgent(config)


@pytest.fixture
def test_agent_b(mock_provider):
    """Create test agent B with different capabilities."""
    
    class MockAgent(BaseAgent):
        async def _execute_internal(self, task):
            return TaskResult(
                task_id=task.id,
                agent_id=self.config.id,
                output=f"Agent B response to: {task.input[:30]}",
                confidence=0.8,
                tokens_used=80,
                processing_time_ms=300,
                status=TaskStatus.COMPLETED,
                metadata={"agent": "B", "mock": True}
            )
    
    config = AgentConfig(
        id="agent_b",
        name="Test Agent B",
        description="Test agent for code generation",
        capabilities=["code_generation", "code_review", "technical_writing"],
        provider="mock",
        model="mock-model",
        system_prompt="You are a coding assistant.",
        max_concurrency=3
    )
    
    return MockAgent(config)


@pytest.fixture
def test_agent_c(mock_provider):
    """Create test agent C that fails."""
    
    class MockAgent(BaseAgent):
        async def _execute_internal(self, task):
            return TaskResult(
                task_id=task.id,
                agent_id=self.config.id,
                output="",
                confidence=0.0,
                tokens_used=0,
                processing_time_ms=100,
                status=TaskStatus.FAILED,
                error="Mock failure",
                metadata={"agent": "C", "mock": True, "failed": True}
            )
    
    config = AgentConfig(
        id="agent_c",
        name="Test Agent C (Failing)",
        description="Test agent that always fails",
        capabilities=["failure_testing"],
        provider="mock",
        model="mock-model",
        max_concurrency=1
    )
    
    return MockAgent(config)


@pytest.fixture
async def registry(test_agent_a, test_agent_b, test_agent_c):
    """Create registry with test agents."""
    registry = AgentRegistry()
    await registry.register(test_agent_a)
    await registry.register(test_agent_b)
    await registry.register(test_agent_c)
    return registry


@pytest.fixture
async def orchestrator(registry):
    """Create orchestrator with test registry."""
    orchestrator = Orchestrator()
    orchestrator.registry = registry  # Replace with test registry
    await orchestrator.start()
    yield orchestrator
    await orchestrator.stop()


@pytest.fixture
def client():
    """Create test client for FastAPI app with pre-initialized orchestrator."""
    # Initialize orchestrator for TestClient (which doesn't trigger lifespan)
    from llm_orchestration_framework.api.server import _orchestrator
    from llm_orchestration_framework.core.orchestrator import Orchestrator
    global_orch = Orchestrator()
    # Use importlib to set the module variable
    import llm_orchestration_framework.api.server as server_mod
    server_mod._orchestrator = global_orch
    
    yield TestClient(app)
    
    # Cleanup
    server_mod._orchestrator = None


@pytest.fixture
def sample_task():
    """Create a sample task."""
    return Task(
        input="Classify this text: The product is great but delivery was slow.",
        metadata={"capabilities": ["classification"]}
    )


@pytest.fixture
def sample_tasks():
    """Create sample tasks for chains."""
    return [
        Task(input="First task: analyze sentiment"),
        Task(input="Second task: generate summary"),
        Task(input="Third task: create response")
    ]


@pytest.fixture
def sample_task_result():
    """Create a sample task result."""
    return TaskResult(
        task_id=uuid.uuid4(),
        agent_id="test_agent",
        output="Test response",
        confidence=0.9,
        tokens_used=150,
        processing_time_ms=750,
        status=TaskStatus.COMPLETED,
        metadata={"test": True}
    )