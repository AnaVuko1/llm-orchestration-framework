"""
DeepSeek provider implementation.
"""

import os
from typing import Any, Dict, Optional

import httpx

from .base import BaseProvider


class DeepSeekProvider(BaseProvider):
    """
    DeepSeek provider for DeepSeek models.
    
    Requires DEEPSEEK_API_KEY environment variable.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "deepseek-chat"
    ):
        """
        Initialize DeepSeek provider.
        
        Args:
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
            base_url: Base URL for DeepSeek API
            default_model: Default model to use
        """
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DeepSeek API key required. "
                "Set DEEPSEEK_API_KEY environment variable or pass api_key parameter."
            )
        
        super().__init__(api_key=api_key, base_url=base_url)
        self.default_model = default_model
        
        # Set default base URL if not provided
        if not self.base_url:
            self.base_url = "https://api.deepseek.com/v1"
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: str = "default",
        **kwargs: Any
    ) -> str:
        """
        Generate a response using DeepSeek API.
        
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
        await self._rate_limit(min_interval_ms=50)
        
        # Prepare messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2000),  # DeepSeek supports longer responses
        }
        
        # Add optional parameters
        optional_params = ["top_p", "frequency_penalty", "presence_penalty", "stream"]
        for param in optional_params:
            if param in kwargs:
                payload[param] = kwargs[param]
        
        url = f"{self.base_url}/chat/completions"
        headers = self._get_auth_headers()
        
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
                    
                    if "choices" not in data or len(data["choices"]) == 0:
                        raise ValueError("No choices in DeepSeek response")
                    
                    self.call_count += 1
                    self.error_count = 0  # Reset error count on success
                    
                    return data["choices"][0]["message"]["content"]
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit
                    retry_count += 1
                    if not await self._exponential_backoff(max_retries=3, base_delay=2.0):
                        raise RuntimeError(f"Rate limit exceeded after {max_retries} retries") from e
                    continue
                elif e.response.status_code == 401:  # Auth error
                    raise ValueError(f"Authentication failed: {e.response.text}") from e
                else:
                    raise RuntimeError(f"DeepSeek API error: {e.response.text}") from e
                    
            except httpx.HTTPError as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError(f"HTTP error after {max_retries} retries: {str(e)}") from e
                await asyncio.sleep(1.0)
                continue
        
        raise RuntimeError(f"Failed after {max_retries} retries")
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for DeepSeek models.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
            
        Note:
            DeepSeek uses a Byte-Pair Encoding (BPE) tokenizer similar to GPT.
            We use a simple approximation: ~4 chars per token for English text.
        """
        # Rough approximation: 1 token ≈ 4 characters for English text
        if not text:
            return 0
        return len(text) // 4
    
    def supports_model(self, model: str) -> bool:
        """
        Check if provider supports a specific model.
        
        Args:
            model: Model name to check
            
        Returns:
            True if model is supported (contains "deepseek")
        """
        return "deepseek" in model.lower()