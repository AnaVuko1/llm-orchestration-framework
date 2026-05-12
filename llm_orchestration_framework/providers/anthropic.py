"""
Anthropic provider implementation.
"""

import os
from typing import Any, Dict, Optional

import httpx

from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    """
    Anthropic provider for Claude models.
    
    Requires ANTHROPIC_API_KEY environment variable.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "claude-3-5-sonnet-20241022"
    ):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            base_url: Base URL for Anthropic API
            default_model: Default model to use
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )
        
        super().__init__(api_key=api_key, base_url=base_url)
        self.default_model = default_model
        
        # Set default base URL if not provided
        if not self.base_url:
            self.base_url = "https://api.anthropic.com/v1"
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: str = "default",
        **kwargs: Any
    ) -> str:
        """
        Generate a response using Anthropic API.
        
        Args:
            prompt: The user prompt
            system: Optional system prompt
            model: Model name (uses default if "default")
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text response
            
        Raises:
            httpx.HTTPError: If API call fails
            ValueError: If response parsing fails
        """
        if model == "default":
            model = self.default_model
        
        # Rate limiting
        await self._rate_limit(min_interval_ms=100)
        
        # Prepare messages (Anthropic uses different format)
        messages = [{"role": "user", "content": prompt}]
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
        }
        
        # Add system prompt if provided
        if system:
            payload["system"] = system
        
        # Add optional parameters
        optional_params = ["top_p", "top_k", "stream"]
        for param in optional_params:
            if param in kwargs:
                payload[param] = kwargs[param]
        
        url = f"{self.base_url}/messages"
        headers = {
            **self._get_auth_headers(),
            "anthropic-version": "2023-06-01",
            "x-api-key": self.api_key,
        }
        
        # Exponential backoff for rate limiting
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        url,
                        headers=headers,
                        json=payload
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    if "content" not in data or len(data["content"]) == 0:
                        raise ValueError("No content in Anthropic response")
                    
                    self.call_count += 1
                    self.error_count = 0  # Reset error count on success
                    
                    # Anthropic returns content as list of content blocks
                    content_blocks = data["content"]
                    text_response = ""
                    for block in content_blocks:
                        if block.get("type") == "text":
                            text_response += block.get("text", "")
                    
                    return text_response
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit
                    retry_count += 1
                    if not await self._exponential_backoff(max_retries=3, base_delay=2.0):
                        raise RuntimeError(f"Rate limit exceeded after {max_retries} retries") from e
                    continue
                elif e.response.status_code == 401:  # Auth error
                    raise ValueError(f"Authentication failed: {e.response.text}") from e
                else:
                    raise RuntimeError(f"Anthropic API error: {e.response.text}") from e
                    
            except httpx.HTTPError as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError(f"HTTP error after {max_retries} retries: {str(e)}") from e
                await asyncio.sleep(1.0)
                continue
        
        raise RuntimeError(f"Failed after {max_retries} retries")
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for Anthropic models.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
            
        Note:
            This is an approximation. Claude uses a different tokenizer than GPT.
            We use a simple approximation: ~3.5 chars per token for English text.
        """
        # Rough approximation: 1 token ≈ 3.5 characters for English text
        # Anthropic's tokenizer is slightly different from OpenAI's
        if not text:
            return 0
        return len(text) // 3.5
    
    def supports_model(self, model: str) -> bool:
        """
        Check if provider supports a specific model.
        
        Args:
            model: Model name to check
            
        Returns:
            True if model is supported (starts with claude-)
        """
        return model.startswith("claude-")