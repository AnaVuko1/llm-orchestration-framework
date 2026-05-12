"""
Chain engine for executing sequences of tasks.
"""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from .agent import BaseAgent
from .registry import AgentRegistry
from .router import TaskRouter
from .task import Task, TaskResult, TaskStatus


class ChainEngine:
    """
    Engine for executing chains of tasks.
    
    Supports:
    - Sequential execution
    - Parallel execution
    - Conditional routing
    - Fallback patterns
    """
    
    def __init__(self, registry: AgentRegistry):
        """
        Initialize the chain engine.
        
        Args:
            registry: Agent registry for routing tasks
        """
        self.registry = registry
        self.router = TaskRouter()
    
    async def execute_sequential(
        self,
        tasks: List[Task],
        context_accumulation: bool = True
    ) -> List[TaskResult]:
        """
        Execute tasks sequentially, optionally accumulating context.
        
        Args:
            tasks: List of tasks to execute in order
            context_accumulation: Whether to pass previous results as context
            
        Returns:
            List of task results in order of execution
        """
        results: List[TaskResult] = []
        accumulated_context: Dict[str, Any] = {}
        
        for i, task in enumerate(tasks):
            # Add accumulated context if enabled and not first task
            if context_accumulation and i > 0 and results:
                task.context = {
                    **task.context,
                    "previous_results": [
                        {
                            "task_id": str(res.task_id),
                            "output": res.output,
                            "status": res.status.value,
                            "confidence": res.confidence
                        }
                        for res in results
                    ],
                    **accumulated_context
                }
            
            # Execute task
            result = await self._execute_single_task(task)
            results.append(result)
            
            # Update accumulated context
            if result.status == TaskStatus.COMPLETED:
                accumulated_context[f"result_{i}"] = result.output[:500]  # Truncate
            
            # Stop on failure if not configured to continue
            if result.status == TaskStatus.FAILED:
                break
        
        return results
    
    async def execute_parallel(
        self,
        tasks: List[Task],
        max_concurrent: Optional[int] = None
    ) -> List[TaskResult]:
        """
        Execute tasks in parallel.
        
        Args:
            tasks: List of tasks to execute concurrently
            max_concurrent: Maximum concurrent tasks (default: len(tasks))
            
        Returns:
            List of task results, order matches input order
        """
        if max_concurrent is None:
            max_concurrent = len(tasks)
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(task: Task) -> TaskResult:
            async with semaphore:
                return await self._execute_single_task(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(
            *[execute_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
        
        # Convert exceptions to FAILED TaskResults
        final_results: List[TaskResult] = []
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                final_results.append(TaskResult(
                    task_id=task.id,
                    agent_id="system",
                    output="",
                    confidence=0.0,
                    tokens_used=0,
                    processing_time_ms=0,
                    status=TaskStatus.FAILED,
                    error=f"Chain execution error: {str(result)}",
                    metadata={"error_type": type(result).__name__}
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def execute_conditional(
        self,
        task: Task,
        condition_fn: Callable[[TaskResult], str]
    ) -> TaskResult:
        """
        Execute a task and route based on condition.
        
        Args:
            task: The initial task to execute
            condition_fn: Function that evaluates result and returns next agent/capability
            
        Returns:
            Final task result after conditional routing
        """
        # Execute initial task
        result = await self._execute_single_task(task)
        
        # Evaluate condition
        next_target = condition_fn(result)
        
        if not next_target:
            return result
        
        # Route to next agent based on condition
        # Check if next_target is an agent ID
        next_agent = await self.registry.get(next_target)
        if next_agent:
            # Direct agent ID
            return await self._execute_with_agent(task, next_agent)
        
        # Otherwise treat as capability
        agents = await self.registry.find_by_capability(next_target)
        if agents:
            # Pick best agent for this capability
            best_agent = max(agents, key=lambda a: a.get_state().success_rate())
            return await self._execute_with_agent(task, best_agent)
        
        # No match found
        return TaskResult(
            task_id=task.id,
            agent_id="system",
            output="",
            confidence=0.0,
            tokens_used=0,
            processing_time_ms=result.processing_time_ms,
            status=TaskStatus.ESCALATED,
            error=f"Condition required agent/capability '{next_target}' but none found",
            metadata={"original_result": result.model_dump()}
        )
    
    async def execute_fallback(
        self,
        task: Task,
        primary_capability: str,
        fallback_capability: str
    ) -> TaskResult:
        """
        Execute task with fallback pattern.
        
        Args:
            task: The task to execute
            primary_capability: Primary capability to try first
            fallback_capability: Fallback capability if primary fails
            
        Returns:
            Task result after fallback attempts
        """
        # Find agents for primary capability
        primary_agents = await self.registry.find_by_capability(primary_capability)
        
        if not primary_agents:
            # Try fallback immediately if no primary agents
            return await self._execute_with_fallback(task, fallback_capability)
        
        # Try primary agents
        for agent in sorted(primary_agents, key=lambda a: a.get_state().success_rate(), reverse=True):
            result = await agent.execute(task)
            if result.status == TaskStatus.COMPLETED:
                return result
        
        # All primary agents failed, try fallback
        return await self._execute_with_fallback(task, fallback_capability)
    
    async def _execute_single_task(self, task: Task) -> TaskResult:
        """
        Execute a single task with routing.
        
        Args:
            task: The task to execute
            
        Returns:
            Task result
        """
        # Route task
        agent, confidence = await self.router.route(task, self.registry)
        
        if not agent:
            # No suitable agent found
            return TaskResult(
                task_id=task.id,
                agent_id="system",
                output="",
                confidence=0.0,
                tokens_used=0,
                processing_time_ms=0,
                status=TaskStatus.ESCALATED,
                error="No suitable agent found for task",
                metadata={"task_capabilities": task.metadata.get("capabilities", [])}
            )
        
        # Execute with routed agent
        result = await agent.execute(task)
        result.confidence = max(result.confidence, confidence)
        
        return result
    
    async def _execute_with_agent(self, task: Task, agent: BaseAgent) -> TaskResult:
        """Execute task with specific agent."""
        return await agent.execute(task)
    
    async def _execute_with_fallback(self, task: Task, fallback_capability: str) -> TaskResult:
        """Execute task with fallback capability."""
        fallback_agents = await self.registry.find_by_capability(fallback_capability)
        
        if not fallback_agents:
            # No fallback agents available
            return TaskResult(
                task_id=task.id,
                agent_id="system",
                output="",
                confidence=0.0,
                tokens_used=0,
                processing_time_ms=0,
                status=TaskStatus.ESCALATED,
                error=f"No agents available for fallback capability '{fallback_capability}'",
                metadata={"task_capabilities": task.metadata.get("capabilities", [])}
            )
        
        # Try all fallback agents
        for agent in sorted(fallback_agents, key=lambda a: a.get_state().success_rate(), reverse=True):
            result = await agent.execute(task)
            if result.status == TaskStatus.COMPLETED:
                return result
        
        # All fallback agents failed
        return TaskResult(
            task_id=task.id,
            agent_id="system",
            output="",
            confidence=0.0,
            tokens_used=0,
            processing_time_ms=0,
            status=TaskStatus.FAILED,
            error=f"All fallback agents failed for capability '{fallback_capability}'",
            metadata={"task_capabilities": task.metadata.get("capabilities", [])}
        )