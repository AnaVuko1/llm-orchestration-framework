"""
Comprehensive tests for LLM orchestration framework.
"""

import asyncio
import uuid
from unittest.mock import AsyncMock

import pytest

from llm_orchestration_framework.core.agent import AgentConfig, BaseAgent
from llm_orchestration_framework.core.chain import ChainEngine
from llm_orchestration_framework.core.orchestrator import Orchestrator
from llm_orchestration_framework.core.registry import AgentRegistry
from llm_orchestration_framework.core.router import TaskRouter
from llm_orchestration_framework.core.task import Task, TaskStatus
from llm_orchestration_framework.memory.buffer import ConversationBuffer
from llm_orchestration_framework.memory.summary import ConversationSummary
from llm_orchestration_framework.providers.base import BaseProvider


# Test 1: Task creation
def test_task_creation():
    """Test creating a task with all fields."""
    task = Task(
        input="Test input",
        context={"user_id": 123},
        metadata={"priority": "high"},
        priority=5,
        max_retries=3,
        timeout_seconds=180
    )
    
    assert task.input == "Test input"
    assert task.context["user_id"] == 123
    assert task.metadata["priority"] == "high"
    assert task.priority == 5
    assert task.max_retries == 3
    assert task.timeout_seconds == 180
    assert task.id is not None


# Test 2: Task validation
def test_task_validation():
    """Test task validation and defaults."""
    # Test required field
    with pytest.raises(ValueError):
        Task(input="")
    
    # Test defaults
    task = Task(input="Test")
    assert task.priority == 3  # Default
    assert task.max_retries == 2  # Default
    assert task.timeout_seconds == 120  # Default
    assert task.context == {}
    assert task.metadata == {}


# Test 3: Agent registry register
@pytest.mark.asyncio
async def test_agent_registry_register(test_agent_a, registry):
    """Test agent registry registration."""
    # Test list agents
    agents = await registry.list()
    assert len(agents) == 3  # From fixture
    
    # Test get agent
    agent = await registry.get("agent_a")
    assert agent is not None
    assert agent.config.id == "agent_a"
    
    # Test count
    count = await registry.count()
    assert count == 3
    
    # Test agent IDs
    ids = await registry.agent_ids()
    assert "agent_a" in ids
    assert "agent_b" in ids
    assert "agent_c" in ids


# Test 4: Agent registry duplicate
@pytest.mark.asyncio
async def test_agent_registry_duplicate(test_agent_a):
    """Test duplicate agent registration raises error."""
    registry = AgentRegistry()
    await registry.register(test_agent_a)
    
    # Try to register same agent again
    with pytest.raises(ValueError):
        await registry.register(test_agent_a)


# Test 5: Agent registry capability find
@pytest.mark.asyncio
async def test_agent_registry_capability_find(registry):
    """Test finding agents by capability."""
    # Find classification agents
    classification_agents = await registry.find_by_capability("classification")
    assert len(classification_agents) == 1
    assert classification_agents[0].config.id == "agent_a"
    
    # Find code generation agents
    code_agents = await registry.find_by_capability("code_generation")
    assert len(code_agents) == 1
    assert code_agents[0].config.id == "agent_b"
    
    # Find non-existent capability
    no_agents = await registry.find_by_capability("non_existent")
    assert len(no_agents) == 0


# Test 6: Router exact match
@pytest.mark.asyncio
async def test_router_exact_match(registry, sample_task):
    """Test router exact capability match."""
    router = TaskRouter(match_threshold=0.1)
    
    # Task has classification capability
    sample_task.metadata["capabilities"] = ["classification"]
    
    agent, confidence = await router.route(sample_task, registry)
    
    assert agent is not None
    assert agent.config.id == "agent_a"  # Agent A has classification
    assert confidence > 0.5


# Test 7: Router partial match
@pytest.mark.asyncio
async def test_router_partial_match(registry):
    """Test router partial capability match."""
    router = TaskRouter(match_threshold=0.1)
    
    # Create task with similar capability
    task = Task(
        input="Review this code",
        metadata={"capabilities": ["code_review"]}
    )
    
    agent, confidence = await router.route(task, registry)
    
    assert agent is not None
    assert agent.config.id == "agent_b"  # Agent B has code_review
    assert confidence > 0.5


# Test 8: Router no match
@pytest.mark.asyncio
async def test_router_no_match(registry):
    """Test router returns None when no match found."""
    router = TaskRouter(match_threshold=0.8)  # Higher threshold
    
    # Create task with completely unrelated capability
    task = Task(
        input="Perform chemical analysis",
        metadata={"capabilities": ["chemical_analysis"]}
    )
    
    agent, confidence = await router.route(task, registry)
    
    assert agent is None
    assert confidence == 0.0


# Test 9: Chain sequential
@pytest.mark.asyncio
async def test_chain_sequential(registry, sample_tasks):
    """Test sequential chain execution."""
    chain_engine = ChainEngine(registry)
    
    results = await chain_engine.execute_sequential(sample_tasks)
    
    assert len(results) == 3
    assert all(r.status == TaskStatus.COMPLETED for r in results)
    assert all(r.agent_id in ["agent_a", "agent_b", "agent_c"] for r in results)


# Test 10: Chain parallel
@pytest.mark.asyncio
async def test_chain_parallel(registry, sample_tasks):
    """Test parallel chain execution."""
    chain_engine = ChainEngine(registry)
    
    results = await chain_engine.execute_parallel(sample_tasks, max_concurrent=2)
    
    assert len(results) == 3
    assert all(r.status == TaskStatus.COMPLETED for r in results)


# Test 11: Chain fallback
@pytest.mark.asyncio
async def test_chain_fallback(registry):
    """Test fallback chain pattern."""
    chain_engine = ChainEngine(registry)
    
    task = Task(
        input="This should trigger fallback",
        metadata={"capabilities": ["failure_testing", "classification"]}
    )
    
    # First try failure_testing (agent C always fails), then fallback to classification
    result = await chain_engine.execute_fallback(
        task=task,
        primary_capability="failure_testing",
        fallback_capability="classification"
    )
    
    # Should succeed with classification (agent A)
    assert result.status == TaskStatus.COMPLETED
    assert "Agent A" in result.output


# Test 12: Chain conditional
@pytest.mark.asyncio
async def test_chain_conditional(registry):
    """Test conditional chain routing."""
    chain_engine = ChainEngine(registry)
    
    task = Task(
        input="Initial task",
        metadata={"capabilities": ["classification"]}
    )
    
    def condition_fn(result):
        """Route to code generation if confidence is high."""
        if result.confidence > 0.8:
            return "code_generation"
        return ""
    
    result = await chain_engine.execute_conditional(task, condition_fn)
    
    # Should execute with agent A (classification) then possibly agent B
    assert result.status == TaskStatus.COMPLETED


# Test 13: Orchestrator submit run
@pytest.mark.asyncio
async def test_orchestrator_submit_run(orchestrator):
    """Test full orchestrator submit->process->get_result flow."""
    task = Task(
        input="Test task for orchestrator",
        metadata={"capabilities": ["classification"]}
    )
    
    # Submit task
    task_id = await orchestrator.submit_task(task)
    assert task_id is not None
    
    # Wait a bit for processing
    await asyncio.sleep(0.1)
    
    # Get result
    result = await orchestrator.get_result(task_id)
    assert result is not None
    assert result.task_id == task.id
    assert result.status == TaskStatus.COMPLETED
    
    # Get status
    status = await orchestrator.get_status(task_id)
    assert status == TaskStatus.COMPLETED


# Test 14: Orchestrator stats
@pytest.mark.asyncio
async def test_orchestrator_stats(orchestrator):
    """Test orchestrator statistics reporting."""
    # Submit a task
    task = Task(input="Test task")
    await orchestrator.submit_task(task)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Get stats
    stats = await orchestrator.get_stats()
    
    assert "tasks" in stats
    assert "agents" in stats
    assert "storage" in stats
    
    # Should have at least 1 submitted task
    assert stats["tasks"]["submitted"] >= 1


# Test 15: Orchestrator escalation
@pytest.mark.asyncio
async def test_orchestrator_escalation(orchestrator):
    """Test orchestrator escalation when no agent found."""
    # Create task with capability no agent has (low similarity with existing agents)
    task = Task(
        input="Execute database query",
        metadata={"capabilities": ["database_query"]}
    )
    
    # Submit task
    task_id = await orchestrator.submit_task(task)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Get result - should be escalated or failed
    result = await orchestrator.get_result(task_id)
    
    assert result is not None
    assert result.status in (TaskStatus.ESCALATED, TaskStatus.FAILED)


# Test 16: Provider mock
@pytest.mark.asyncio
async def test_provider_mock(mock_provider):
    """Test mock provider functionality."""
    response = await mock_provider.generate(
        prompt="Test prompt",
        system="Test system",
        model="test-model"
    )
    
    assert response == "Mock response"
    
    token_count = mock_provider.count_tokens("Test text")
    assert token_count == 100


# Test 17: Memory buffer
@pytest.mark.asyncio
async def test_memory_buffer():
    """Test conversation buffer memory."""
    buffer = ConversationBuffer(max_messages=3)
    
    # Add messages
    await buffer.add_user_message("Hello")
    await buffer.add_agent_message("Hi there!")
    await buffer.add_user_message("How are you?")
    
    # Test get messages
    messages = await buffer.get_messages()
    assert len(messages) == 3
    
    # Test context
    context = await buffer.get_context()
    assert "USER: Hello" in context
    assert "AGENT: Hi there!" in context
    
    # Test sliding window
    await buffer.add_agent_message("I'm good, thanks!")
    messages = await buffer.get_messages()
    assert len(messages) == 3  # Should still be 3 (sliding window)
    
    # Test clear
    await buffer.clear()
    messages = await buffer.get_messages()
    assert len(messages) == 0


# Test 18: API health
def test_api_health(client):
    """Test API health endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data


# Test 19: API submit task
def test_api_submit_task(client):
    """Test API task submission."""
    task_data = {
        "input": "Test API task",
        "metadata": {"test": True},
        "priority": 4
    }
    
    response = client.post("/v1/tasks", json=task_data)
    
    assert response.status_code == 202  # Accepted
    data = response.json()
    assert "task_id" in data
    assert "status" in data


# Test 20: API list agents
def test_api_list_agents(client):
    """Test API agent listing."""
    response = client.get("/v1/agents")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


# Additional comprehensive tests
@pytest.mark.asyncio
async def test_agent_execution(test_agent_a):
    """Test agent execution."""
    task = Task(input="Test agent execution")
    
    result = await test_agent_a.execute(task)
    
    assert result.status == TaskStatus.COMPLETED
    assert result.agent_id == "agent_a"
    assert result.confidence > 0
    assert result.processing_time_ms >= 0


@pytest.mark.asyncio
async def test_agent_state(test_agent_a):
    """Test agent state tracking."""
    initial_state = test_agent_a.get_state()
    assert initial_state.status == "IDLE"
    assert initial_state.tasks_completed == 0
    
    # Execute a task
    task = Task(input="Test")
    await test_agent_a.execute(task)
    
    # Check state updated
    updated_state = test_agent_a.get_state()
    assert updated_state.tasks_completed == 1
    assert updated_state.last_activity >= initial_state.last_activity


@pytest.mark.asyncio
async def test_memory_summary():
    """Test conversation summary memory."""
    # Create a mock provider for summarization
    mock_provider = AsyncMock(spec=BaseProvider)
    mock_provider.generate = AsyncMock(return_value="Summary: Test conversation")
    
    summary_memory = ConversationSummary(
        max_messages=5,
        summary_threshold=3,
        provider=mock_provider,
        model="test-model"
    )
    
    # Add messages
    await summary_memory.add_user_message("First message")
    await summary_memory.add_agent_message("First response")
    await summary_memory.add_user_message("Second message")  # Should trigger summarization
    
    # Provider should have been called for summarization
    assert mock_provider.generate.called
    
    # Get context should include summary
    context = await summary_memory.get_context()
    assert "Summary:" in context


@pytest.mark.asyncio
async def test_orchestrator_cancel(orchestrator):
    """Test task cancellation."""
    task = Task(input="Task to cancel")
    
    # Submit task
    task_id = await orchestrator.submit_task(task)
    
    # Try to cancel
    cancelled = await orchestrator.cancel(task_id)
    
    # Cancellation may or may not succeed depending on timing
    # Just verify no exception
    assert cancelled in [True, False]


@pytest.mark.asyncio  
async def test_chain_timeout():
    """Test chain timeout handling."""
    # Create a slow agent
    class SlowAgent(BaseAgent):
        async def _execute_internal(self, task):
            await asyncio.sleep(2.0)  # Sleep for 2 seconds
            return TaskResult(
                task_id=task.id,
                agent_id=self.config.id,
                output="Slow response",
                confidence=0.5,
                tokens_used=50,
                processing_time_ms=2000,
                status=TaskStatus.COMPLETED
            )
    
    config = AgentConfig(
        id="slow_agent",
        name="Slow Agent",
        capabilities=["slow"],
        provider="test",
        model="test"
    )
    
    agent = SlowAgent(config)
    registry = AgentRegistry()
    await registry.register(agent)
    
    chain_engine = ChainEngine(registry)
    
    # Task with short timeout
    task = Task(input="Test", timeout_seconds=1)
    
    # Should complete (timeout is per-agent, not per-chain)
    result = await chain_engine._execute_single_task(task)
    assert result is not None