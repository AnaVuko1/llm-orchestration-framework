"""
Conversation summary memory with automatic summarization.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from ..providers.base import BaseProvider
from .base import BaseMemory


class ConversationSummary(BaseMemory):
    """
    Conversation memory that automatically summarizes when buffer exceeds threshold.
    
    Uses a provider to generate summaries recursively.
    
    Attributes:
        max_messages: Maximum messages in buffer before summarization
        summary_threshold: Trigger summarization when buffer reaches this size
        provider: Provider to use for summarization
        model: Model to use for summarization
        buffer: Internal buffer for recent messages
        summaries: List of historical summaries
    """
    
    def __init__(
        self,
        max_messages: int = 50,
        summary_threshold: int = 20,
        provider: Optional[BaseProvider] = None,
        model: str = "default"
    ):
        """
        Initialize conversation summary memory.
        
        Args:
            max_messages: Maximum messages to keep in buffer
            summary_threshold: Trigger summarization when buffer reaches this size
            provider: Provider to use for summarization
            model: Model to use for summarization
        """
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        self.provider = provider
        self.model = model
        
        self.buffer: List[Dict[str, Any]] = []
        self.summaries: List[str] = []
    
    async def add(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a message to memory.
        
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
        
        self.buffer.append(message)
        
        # Check if we need to summarize
        if len(self.buffer) >= self.summary_threshold and self.provider:
            await self._summarize_buffer()
        
        # Enforce maximum buffer size
        if len(self.buffer) > self.max_messages:
            self.buffer = self.buffer[-self.max_messages:]
    
    async def get_context(self, task_id: Optional[str] = None) -> str:
        """
        Get conversation context including summaries and recent messages.
        
        Args:
            task_id: Optional task ID to filter messages
            
        Returns:
            Formatted context string
        """
        parts = []
        
        # Add historical summaries
        if self.summaries:
            summary_text = "\n\n".join(self.summaries)
            parts.append(f"CONVERSATION HISTORY (SUMMARIZED):\n{summary_text}")
        
        # Add recent messages from buffer
        if self.buffer:
            recent_messages = self._format_messages(self.buffer)
            parts.append(f"RECENT CONVERSATION:\n{recent_messages}")
        
        return "\n\n" + "\n\n".join(parts) if parts else ""
    
    async def clear(self) -> None:
        """Clear all messages and summaries."""
        self.buffer.clear()
        self.summaries.clear()
    
    async def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get raw message history.
        
        Args:
            limit: Optional limit on number of messages to return
            
        Returns:
            List of message dictionaries
        """
        # Return buffer messages (historical messages are summarized)
        messages = self.buffer.copy()
        if limit is not None and limit > 0:
            messages = messages[-limit:]
        return messages
    
    async def _summarize_buffer(self) -> None:
        """Summarize the current buffer and add to summaries."""
        if not self.buffer or not self.provider:
            return
        
        # Prepare text to summarize
        conversation_text = self._format_messages(self.buffer)
        
        # Generate summary
        try:
            summary = await self.provider.generate(
                prompt=f"Summarize the following conversation concisely, preserving key points, decisions, and important context:\n\n{conversation_text}",
                system="You are a conversation summarizer. Create a concise summary that captures the essence of the conversation, including key decisions, action items, and important context.",
                model=self.model
            )
            
            # Add to summaries
            self.summaries.append(summary.strip())
            
            # Clear buffer (messages are now summarized)
            self.buffer.clear()
            
            # Summarize summaries if we have too many
            if len(self.summaries) > 5:
                await self._summarize_summaries()
                
        except Exception as e:
            # Log error but continue (don't crash on summarization failure)
            print(f"Summarization failed: {e}")
            # Keep buffer messages instead of losing them
            # Maybe keep buffer but mark as not to be summarized again immediately
    
    async def _summarize_summaries(self) -> None:
        """Recursively summarize existing summaries."""
        if len(self.summaries) <= 3 or not self.provider:
            return
        
        # Combine recent summaries
        recent_summaries = self.summaries[-5:]  # Last 5 summaries
        combined = "\n\n---\n\n".join(recent_summaries)
        
        try:
            meta_summary = await self.provider.generate(
                prompt=f"Combine and summarize the following conversation summaries into a single concise summary:\n\n{combined}",
                system="You are a meta-summarizer. Combine multiple conversation summaries into one coherent summary that captures the overall conversation flow and key points.",
                model=self.model
            )
            
            # Replace recent summaries with meta-summary
            self.summaries = self.summaries[:-5] + [meta_summary.strip()]
            
        except Exception as e:
            # Log error but continue
            print(f"Meta-summarization failed: {e}")
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages for display or summarization."""
        formatted = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)
    
    async def get_summary_count(self) -> int:
        """Get number of stored summaries."""
        return len(self.summaries)
    
    async def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()