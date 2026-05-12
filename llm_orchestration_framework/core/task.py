"""
Task models for LLM orchestration.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class TaskStatus(str, Enum):
    """Status of a task in the orchestration system."""
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ESCALATED = "ESCALATED"
    CANCELLED = "CANCELLED"


class Task(BaseModel):
    """
    A task to be executed by an agent.
    
    Attributes:
        id: Unique identifier (auto-generated if not provided)
        agent_id: ID of agent assigned to task (optional)
        input: The input text for the task
        context: Additional context as a dictionary
        metadata: Arbitrary metadata
        parent_task_id: ID of parent task if this is part of a chain
        chain_id: ID of the chain this task belongs to
        priority: Priority from 1 (lowest) to 5 (highest)
        max_retries: Maximum number of retry attempts
        timeout_seconds: Timeout in seconds
    """
    id: UUID = Field(default_factory=uuid.uuid4)
    agent_id: Optional[str] = None
    input: str = Field(..., min_length=1, description="Task input text")
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parent_task_id: Optional[UUID] = None
    chain_id: Optional[str] = None
    priority: int = Field(default=3, ge=1, le=5)
    max_retries: int = Field(default=2, ge=0)
    timeout_seconds: int = Field(default=120, ge=1)
    
    @field_validator('input')
    @classmethod
    def validate_input_not_empty(cls, v: str) -> str:
        """Ensure input is not empty."""
        if not v or not v.strip():
            raise ValueError("Input cannot be empty")
        return v.strip()


class TaskResult(BaseModel):
    """
    Result of a task execution.
    
    Attributes:
        task_id: ID of the task this result belongs to
        agent_id: ID of agent that executed the task
        output: The output text from the agent
        confidence: Confidence score from 0.0 to 1.0
        tokens_used: Number of tokens used
        processing_time_ms: Processing time in milliseconds
        status: Final status of the task
        error: Error message if task failed
        metadata: Additional metadata about the execution
    """
    task_id: UUID
    agent_id: str
    output: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    tokens_used: int = Field(default=0, ge=0)
    processing_time_ms: int = Field(default=0, ge=0)
    status: TaskStatus
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is within bounds."""
        return min(1.0, max(0.0, v))