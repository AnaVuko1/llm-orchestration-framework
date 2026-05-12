"""
Base provider abstract class for LLM providers.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    Subclasses must implement generate and count_tokens methods.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the provider.
        
        Args:
            api_key: API key for the provider
            base_url: Base URL for API calls (for self-hosted models)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.last_call_time = 0
        self.call_count = 0
        self.error_count = 0
        
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: str = "default",
        **kwargs: Any
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt
            system: Optional system prompt
            model: Model name to use
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If generation fails
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        pass
    
    async def _rate_limit(self, min_interval_ms: int = 100) -> None:
        """
        Simple rate limiting to avoid hitting API limits.
        
        Args:
            min_interval_ms: Minimum milliseconds between calls
        """
        current_time = time.time() * 1000  # Convert to ms
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < min_interval_ms:
            wait_time = (min_interval_ms - time_since_last_call) / 1000
            await asyncio.sleep(wait_time)
        
        self.last_call_time = time.time() * 1000
    
    async def _exponential_backoff(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> bool:
        """
        Exponential backoff for rate limiting errors.
        
        Args:
            max_retries: Maximum number of retries
            base_delay: Base delay in seconds
            
        Returns:
            True if should retry, False if max retries exceeded
        """
        if self.error_count >= max_retries:
            return False
        
        delay = base_delay * (2 ** self.error_count)
        self.error_count += 1
        await asyncio.sleep(delay)
        return True
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for API calls.
        
        Returns:
            Dictionary of headers
        """
        if not self.api_key:
            return {}
        
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def reset_stats(self) -> None:
        """Reset call statistics."""
        self.last_call_time = 0
        self.call_count = 0
        self.error_count = 0