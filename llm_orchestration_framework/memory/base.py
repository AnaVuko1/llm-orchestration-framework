"""
Base memory abstract class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseMemory(ABC):
    """
    Abstract base class for memory systems.
    
    Memory systems store conversation history and provide context for tasks.
    """
    
    @abstractmethod
    async def add(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a message to memory.
        
        Args:
            role: Role of the speaker (e.g., "user", "agent", "system")
            content: Message content
            metadata: Optional metadata about the message
        """
        pass
    
    @abstractmethod
    async def get_context(self, task_id: Optional[str] = None) -> str:
        """
        Get conversation context as a string.
        
        Args:
            task_id: Optional task ID to filter context
            
        Returns:
            Formatted context string
        """
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all messages from memory."""
        pass
    
    @abstractmethod
    async def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get raw message history.
        
        Args:
            limit: Optional limit on number of messages to return
            
        Returns:
            List of message dictionaries with role, content, metadata
        """
        pass
    
    async def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a user message."""
        await self.add("user", content, metadata)
    
    async def add_agent_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an agent message."""
        await self.add("agent", content, metadata)
    
    async def add_system_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a system message."""
        await self.add("system", content, metadata)