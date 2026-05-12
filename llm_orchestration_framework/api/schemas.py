"""
Pydantic schemas for API requests and responses.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from ..core.task import TaskStatus


# Task schemas
class TaskCreate(BaseModel):
    """Schema for creating a task."""
    input: str = Field(..., min_length=1, description="Task input text")
    agent_id: Optional[str] = Field(None, description="Optional specific agent ID")
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parent_task_id: Optional[UUID] = None
    chain_id: Optional[str] = None
    priority: int = Field(default=3, ge=1, le=5)
    max_retries: int = Field(default=2, ge=0)
    timeout_seconds: int = Field(default=120, ge=1)


class TaskResponse(BaseModel):
    """Schema for task response."""
    task_id: UUID
    status: TaskStatus
    created_at: Optional[str] = None


class TaskResultResponse(BaseModel):
    """Schema for task result."""
    task_id: UUID
    agent_id: str
    output: str
    confidence: float
    tokens_used: int
    processing_time_ms: int
    status: TaskStatus
    error: Optional[str] = None
    metadata: Dict[str, Any]


# Agent schemas
class AgentConfigCreate(BaseModel):
    """Schema for creating agent configuration."""
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


class AgentStateResponse(BaseModel):
    """Schema for agent state."""
    agent_id: str
    status: str
    current_task_id: Optional[UUID] = None
    tasks_completed: int
    tasks_failed: int
    last_activity: str
    success_rate: float


class AgentResponse(BaseModel):
    """Schema for agent details."""
    config: AgentConfigCreate
    state: AgentStateResponse


# Chain schemas
class ChainTask(BaseModel):
    """Schema for a task in a chain."""
    input: str = Field(..., min_length=1)
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=3, ge=1, le=5)
    max_retries: int = Field(default=2, ge=0)
    timeout_seconds: int = Field(default=120, ge=1)


class SequentialChainRequest(BaseModel):
    """Schema for sequential chain request."""
    tasks: List[ChainTask] = Field(..., min_items=1)
    context_accumulation: bool = Field(default=True)


class ParallelChainRequest(BaseModel):
    """Schema for parallel chain request."""
    tasks: List[ChainTask] = Field(..., min_items=1)
    max_concurrent: Optional[int] = None


class ChainResponse(BaseModel):
    """Schema for chain response."""
    chain_id: str
    results: List[TaskResultResponse]


# Stats schemas
class TaskStats(BaseModel):
    """Schema for task statistics."""
    submitted: int
    completed: int
    failed: int
    escalated: int
    success_rate: float
    active_tasks: int


class AgentStats(BaseModel):
    """Schema for agent statistics."""
    count: int
    details: List[AgentStateResponse]


class StorageStats(BaseModel):
    """Schema for storage statistics."""
    tasks_stored: int
    results_stored: int
    max_results: int
    cleanup_interval_hours: int


class StatsResponse(BaseModel):
    """Schema for statistics response."""
    tasks: TaskStats
    agents: AgentStats
    storage: StorageStats


# Error schemas
class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error: str
    detail: Optional[str] = None
    code: str = "INTERNAL_ERROR"


# Health schemas
class HealthResponse(BaseModel):
    """Schema for health check."""
    status: str
    timestamp: str
    version: str
    agents_registered: int
    tasks_queued: int