"""
Agent models and base agent implementation.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .task import Task, TaskResult, TaskStatus


class AgentConfig(BaseModel):
    """
    Configuration for an agent.
    
    Attributes:
        id: Unique identifier for the agent
        name: Human-readable name
        description: Agent description
        capabilities: List of capabilities this agent can handle
        provider: Provider name (e.g., "openai", "anthropic")
        model: Model name (e.g., "gpt-4", "claude-3")
        system_prompt: Optional system prompt to use with this agent
        max_concurrency: Maximum concurrent tasks this agent can handle
        timeout_s: Timeout in seconds for agent execution
        metadata: Additional configuration metadata
    """
    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str = ""
    capabilities: List[str] = Field(default_factory=list)
    provider: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    system_prompt: Optional[str] = None
    max_concurrency: int = Field(default=5, ge=1)
    timeout_s: int = Field(default=120, ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class AgentState:
    """
    Runtime state of an agent.
    
    Attributes:
        agent_id: ID of the agent
        status: Current status (IDLE, BUSY, ERROR)
        current_task_id: ID of currently executing task, if any
        tasks_completed: Count of successfully completed tasks
        tasks_failed: Count of failed tasks
        last_activity: Timestamp of last activity
    """
    agent_id: str
    status: str = "IDLE"  # "IDLE", "BUSY", "ERROR"
    current_task_id: Optional[UUID] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    
    def success_rate(self) -> float:
        """Calculate success rate (completed / (completed + failed + 1))."""
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 0.0
        return self.tasks_completed / total
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now()


class BaseAgent:
    """
    Base class for all agents.
    
    Subclasses must implement the execute method.
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.state = AgentState(agent_id=config.id)
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
    
    async def execute(self, task: Task) -> TaskResult:
        """
        Execute a task.
        
        Args:
            task: The task to execute
            
        Returns:
            TaskResult with execution outcome
        """
        start_time = datetime.now()
        self.state.status = "BUSY"
        self.state.current_task_id = task.id
        self.state.update_activity()
        
        try:
            async with self._semaphore:
                # Record processing start
                processing_start = datetime.now()
                
                # Execute the actual task (to be implemented by subclasses)
                result = await self._execute_internal(task)
                
                # Calculate processing time
                processing_time_ms = int(
                    (datetime.now() - processing_start).total_seconds() * 1000
                )
                
                # Update result with timing
                result.processing_time_ms = processing_time_ms
                
                # Update agent state
                if result.status == TaskStatus.COMPLETED:
                    self.state.tasks_completed += 1
                elif result.status == TaskStatus.FAILED:
                    self.state.tasks_failed += 1
                    
        except Exception as e:
            # Fallback result for unexpected errors
            result = TaskResult(
                task_id=task.id,
                agent_id=self.config.id,
                output="",
                confidence=0.0,
                tokens_used=0,
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                status=TaskStatus.FAILED,
                error=f"Unexpected error: {str(e)}",
                metadata={"error_type": type(e).__name__}
            )
            self.state.tasks_failed += 1
            self.state.status = "ERROR"
            
        finally:
            self.state.status = "IDLE"
            self.state.current_task_id = None
            self.state.update_activity()
            
        return result
    
    async def _execute_internal(self, task: Task) -> TaskResult:
        """
        Internal execution method to be implemented by subclasses.
        
        Args:
            task: The task to execute
            
        Returns:
            TaskResult with execution outcome
        """
        raise NotImplementedError("Subclasses must implement _execute_internal")
    
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this agent can handle."""
        return self.config.capabilities.copy()
    
    def get_state(self) -> AgentState:
        """Get current agent state."""
        return self.state
    
    async def reset(self) -> None:
        """Reset agent state."""
        self.state = AgentState(agent_id=self.config.id)
        self.state.update_activity()
    
    def __str__(self) -> str:
        return f"{self.config.name} ({self.config.id})"