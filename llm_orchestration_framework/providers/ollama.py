"""
Ollama provider implementation for local models.
"""

import os
from typing import Any, Dict, Optional

import httpx

from .base import BaseProvider


class OllamaProvider(BaseProvider):
    """
    Ollama provider for local models.
    
    Requires OLLAMA_HOST environment variable or runs on localhost:11434.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "llama3.2:3b"
    ):
        """
        Initialize Ollama provider.
        
        Args:
            api_key: Not used for Ollama (ignored)
            base_url: Base URL for Ollama API (defaults to OLLAMA_HOST or localhost:11434)
            default_model: Default model to use
        """
        # Ollama doesn't use API keys
        super().__init__(api_key=None, base_url=base_url)
        self.default_model = default_model
        
        # Set default base URL if not provided
        if not self.base_url:
            self.base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: str = "default",
        **kwargs: Any
    ) -> str:
        """
        Generate a response using Ollama API.
        
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
        
        # Rate limiting (less strict for local models)
        await self._rate_limit(min_interval_ms=10)
        
        # Prepare messages - Ollama uses a different format
        # We'll use the simple /api/generate endpoint
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 1000),
            }
        }
        
        # Add system prompt if provided
        if system:
            payload["system"] = system
        
        # Add optional parameters to options
        optional_params = ["top_p", "top_k", "repeat_penalty"]
        for param in optional_params:
            if param in kwargs:
                payload["options"][param] = kwargs[param]
        
        url = f"{self.base_url}/api/generate"
        
        # Exponential backoff for connection errors
        max_retries = 5  # More retries for local connection issues
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:  # Longer timeout for local models
                    response = await client.post(
                        url,
                        json=payload
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    if "response" not in data:
                        raise ValueError("No response in Ollama output")
                    
                    self.call_count += 1
                    self.error_count = 0  # Reset error count on success
                    
                    return data["response"]
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:  # Model not found
                    raise ValueError(f"Model '{model}' not found. "
                                   f"Make sure it's pulled with 'ollama pull {model}'") from e
                elif e.response.status_code == 429:  # Rate limit (unlikely for local)
                    retry_count += 1
                    await asyncio.sleep(0.5)
                    continue
                else:
                    # Try to get error message
                    try:
                        error_msg = e.response.json().get("error", str(e))
                    except:
                        error_msg = str(e)
                    raise RuntimeError(f"Ollama API error: {error_msg}") from e
                    
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError(
                        f"Cannot connect to Ollama at {self.base_url}. "
                        f"Is Ollama running? Error: {str(e)}"
                    ) from e
                await asyncio.sleep(1.0)
                continue
                    
            except httpx.HTTPError as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError(f"HTTP error after {max_retries} retries: {str(e)}") from e
                await asyncio.sleep(1.0)
                continue
        
        raise RuntimeError(f"Failed after {max_retries} retries")
    
    async def list_models(self) -> list:
        """
        List available Ollama models.
        
        Returns:
            List of model names
            
        Raises:
            RuntimeError: If cannot connect to Ollama
        """
        url = f"{self.base_url}/api/tags"
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                return models
                
        except httpx.HTTPError as e:
            raise RuntimeError(f"Cannot list Ollama models: {str(e)}") from e
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for Ollama models.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
            
        Note:
            Ollama models use various tokenizers. We use a general approximation.
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
            True if model is available locally (always returns True for Ollama)
            
        Note:
            We can't verify model existence without making an API call,
            so we assume all models are supported.
        """
        return True  # Ollama can use any model name