"""
Conversation buffer with sliding window.
"""

from typing import Any, Dict, List, Optional

from .base import BaseMemory


class ConversationBuffer(BaseMemory):
    """
    Conversation memory that keeps a sliding window of recent messages.
    
    Attributes:
        max_messages: Maximum number of messages to keep
        messages: List of stored messages
    """
    
    def __init__(self, max_messages: int = 20):
        """
        Initialize conversation buffer.
        
        Args:
            max_messages: Maximum number of messages to keep in buffer
        """
        self.max_messages = max_messages
        self.messages: List[Dict[str, Any]] = []
    
    async def add(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a message to the buffer.
        
        Args:
            role: Role of the speaker
            content: Message content
            metadata: Optional metadata
        """
        message = {
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": self._get_timestamp()
        }
        
        self.messages.append(message)
        
        # Enforce sliding window
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    async def get_context(self, task_id: Optional[str] = None) -> str:
        """
        Get conversation context as a formatted string.
        
        Args:
            task_id: Optional task ID to filter messages (ignored in buffer)
            
        Returns:
            Formatted context string
        """
        if not self.messages:
            return ""
        
        # Format messages
        formatted = []
        for msg in self.messages:
            role = msg["role"].upper()
            content = msg["content"]
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    async def clear(self) -> None:
        """Clear all messages from the buffer."""
        self.messages.clear()
    
    async def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get raw message history.
        
        Args:
            limit: Optional limit on number of messages to return
            
        Returns:
            List of message dictionaries
        """
        messages = self.messages.copy()
        if limit is not None and limit > 0:
            messages = messages[-limit:]
        return messages
    
    async def get_message_count(self) -> int:
        """Get current number of messages in buffer."""
        return len(self.messages)
    
    async def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.messages) >= self.max_messages
    
    async def get_recent_messages(self, count: int) -> List[Dict[str, Any]]:
        """
        Get recent messages.
        
        Args:
            count: Number of recent messages to get
            
        Returns:
            List of recent message dictionaries
        """
        if count <= 0:
            return []
        return self.messages[-count:]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()