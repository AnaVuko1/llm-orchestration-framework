"""
Task router for matching tasks to agents based on capabilities.
"""

import difflib
from typing import List, Optional, Tuple

from .agent import BaseAgent
from .registry import AgentRegistry
from .task import Task


class TaskRouter:
    """
    Routes tasks to agents based on capability matching.
    
    Uses three levels of matching:
    1. Exact match
    2. Partial match (Levenshtein distance)
    3. Fuzzy match (difflib)
    
    Confidence scoring:
    match_quality × agent_health
    """
    
    def __init__(self, match_threshold: float = 0.3):
        """
        Initialize the router.
        
        Args:
            match_threshold: Minimum confidence score (0.0-1.0) for a match
        """
        self.match_threshold = match_threshold
    
    def set_threshold(self, threshold: float) -> None:
        """Update match threshold."""
        self.match_threshold = threshold
    
    async def route(self, task: Task, registry: AgentRegistry) -> Tuple[Optional[BaseAgent], float]:
        """
        Route a task to the best available agent.
        
        Args:
            task: The task to route
            registry: Agent registry to search
            
        Returns:
            Tuple of (agent, confidence_score) or (None, 0.0) if no match found
        """
        # Extract capabilities from task input or metadata
        task_capabilities = self._extract_capabilities(task)
        
        if not task_capabilities:
            # No capabilities specified, route to any available agent
            return await self._route_to_any(task, registry)
        
        # Get all agents
        agents = await registry.list()
        if not agents:
            return None, 0.0
        
        best_match = None
        best_confidence = 0.0
        
        for agent in agents:
            agent_capabilities = agent.get_capabilities()
            confidence = self._calculate_match_confidence(
                task_capabilities, agent_capabilities, agent
            )
            
            if confidence > best_confidence:
                best_match = agent
                best_confidence = confidence
        
        # Check if best match meets threshold
        if best_confidence >= self.match_threshold:
            return best_match, best_confidence
        else:
            return None, 0.0
    
    def _extract_capabilities(self, task: Task) -> List[str]:
        """
        Extract capabilities from a task.
        
        Args:
            task: The task to extract capabilities from
            
        Returns:
            List of capability strings
        """
        # Check for explicit capabilities in metadata
        if "capabilities" in task.metadata and isinstance(task.metadata["capabilities"], list):
            return [str(cap).lower() for cap in task.metadata["capabilities"]]
        
        # Extract from input (simple keyword matching)
        # This is a basic implementation - could be enhanced with NLP
        capabilities = []
        input_lower = task.input.lower()
        
        # Simple keyword matching
        common_capabilities = [
            "classification", "summarization", "translation", 
            "code_generation", "code_review", "data_analysis",
            "creative_writing", "technical_writing", "qa",
            "sentiment_analysis", "entity_extraction"
        ]
        
        for cap in common_capabilities:
            if cap in input_lower:
                capabilities.append(cap)
        
        # If no capabilities found, use a default
        if not capabilities:
            capabilities = ["general"]
        
        return capabilities
    
    def _calculate_match_confidence(
        self, task_capabilities: List[str], agent_capabilities: List[str], agent: BaseAgent
    ) -> float:
        """
        Calculate match confidence between task and agent.
        
        Args:
            task_capabilities: Capabilities required by the task
            agent_capabilities: Capabilities provided by the agent
            agent: The agent instance
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        if not task_capabilities or not agent_capabilities:
            return 0.0
        
        # Calculate match quality
        match_quality = self._calculate_match_quality(task_capabilities, agent_capabilities)
        
        # Calculate agent health
        agent_state = agent.get_state()
        agent_health = agent_state.success_rate()
        
        # Combine with slight preference for healthy agents
        confidence = match_quality * (0.7 + 0.3 * agent_health)
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def _calculate_match_quality(self, task_caps: List[str], agent_caps: List[str]) -> float:
        """
        Calculate quality of match between task and agent capabilities.
        
        Args:
            task_caps: Task capabilities
            agent_caps: Agent capabilities
            
        Returns:
            Match quality from 0.0 to 1.0
        """
        if not task_caps or not agent_caps:
            return 0.0
        
        # Try exact matches first
        exact_matches = set(task_caps) & set(agent_caps)
        if exact_matches:
            return 1.0
        
        # Try partial matches (Levenshtein)
        best_partial_match = 0.0
        for task_cap in task_caps:
            for agent_cap in agent_caps:
                similarity = difflib.SequenceMatcher(None, task_cap, agent_cap).ratio()
                if similarity > best_partial_match:
                    best_partial_match = similarity
        
        return best_partial_match
    
    async def _route_to_any(self, task: Task, registry: AgentRegistry) -> Tuple[Optional[BaseAgent], float]:
        """
        Route to any available agent when no capabilities specified.
        
        Args:
            task: The task to route
            registry: Agent registry
            
        Returns:
            Tuple of (agent, confidence_score)
        """
        agents = await registry.list()
        if not agents:
            return None, 0.0
        
        # Find idle agents first
        idle_agents = []
        for agent in agents:
            state = agent.get_state()
            if state.status == "IDLE":
                idle_agents.append(agent)
        
        if idle_agents:
            # Pick the agent with highest success rate
            best_agent = max(idle_agents, key=lambda a: a.get_state().success_rate())
            return best_agent, 0.5  # Medium confidence for generic routing
        
        # If no idle agents, pick any agent
        return agents[0], 0.3  # Low confidence for busy agent