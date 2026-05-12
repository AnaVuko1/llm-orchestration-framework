"""
Agent registry for managing and discovering agents.
"""

import asyncio
from typing import Dict, List, Optional

from .agent import BaseAgent


class AgentRegistry:
    """
    Thread-safe registry for managing agents.
    
    Attributes:
        agents: Dictionary mapping agent_id to agent instance
        _lock: Asyncio lock for thread-safe operations
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, agent: BaseAgent) -> None:
        """
        Register an agent with the registry.
        
        Args:
            agent: The agent to register
            
        Raises:
            ValueError: If agent with same ID already exists
        """
        async with self._lock:
            agent_id = agent.config.id
            if agent_id in self.agents:
                raise ValueError(f"Agent with ID '{agent_id}' already registered")
            self.agents[agent_id] = agent
    
    async def unregister(self, agent_id: str) -> bool:
        """
        Unregister an agent from the registry.
        
        Args:
            agent_id: ID of agent to unregister
            
        Returns:
            True if agent was removed, False if not found
        """
        async with self._lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                return True
            return False
    
    async def get(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: ID of agent to retrieve
            
        Returns:
            Agent instance, or None if not found
        """
        async with self._lock:
            return self.agents.get(agent_id)
    
    async def find_by_capability(self, capability: str) -> List[BaseAgent]:
        """
        Find all agents that have a specific capability.
        
        Args:
            capability: Capability to search for
            
        Returns:
            List of agents with the capability
        """
        async with self._lock:
            matching_agents = []
            for agent in self.agents.values():
                if capability in agent.get_capabilities():
                    matching_agents.append(agent)
            return matching_agents
    
    async def list(self) -> List[BaseAgent]:
        """
        List all registered agents.
        
        Returns:
            List of all agents
        """
        async with self._lock:
            return list(self.agents.values())
    
    async def agent_ids(self) -> List[str]:
        """
        Get all registered agent IDs.
        
        Returns:
            List of agent IDs
        """
        async with self._lock:
            return list(self.agents.keys())
    
    async def count(self) -> int:
        """
        Get number of registered agents.
        
        Returns:
            Count of agents
        """
        async with self._lock:
            return len(self.agents)
    
    async def clear(self) -> None:
        """Clear all registered agents."""
        async with self._lock:
            self.agents.clear()