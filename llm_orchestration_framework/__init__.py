"""
LLM Orchestration Framework

A general-purpose LLM orchestration layer — define agents, chain them together, 
route tasks, handle escalation. Reusable across any domain.
"""

from .core.agent import AgentConfig, BaseAgent, AgentState
from .core.chain import ChainEngine
from .core.orchestrator import Orchestrator
from .core.registry import AgentRegistry
from .core.router import TaskRouter
from .core.task import Task, TaskResult, TaskStatus
from .memory.buffer import ConversationBuffer
from .memory.summary import ConversationSummary

__version__ = "0.1.0"

__all__ = [
    # Core
    "Task",
    "TaskResult", 
    "TaskStatus",
    "AgentConfig",
    "BaseAgent",
    "AgentState",
    "AgentRegistry",
    "TaskRouter",
    "ChainEngine",
    "Orchestrator",
    
    # Memory
    "ConversationBuffer",
    "ConversationSummary",
    
    # Version
    "__version__",
]