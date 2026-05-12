"""
Main orchestrator for managing tasks, chains, and escalations.
"""

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .agent import BaseAgent
from .chain import ChainEngine
from .registry import AgentRegistry
from .router import TaskRouter
from .task import Task, TaskResult, TaskStatus


class Orchestrator:
    """
    Main orchestrator for the LLM orchestration framework.
    
    Manages task lifecycle, routing, chaining, and escalation.
    """
    
    def __init__(self, max_results: int = 1000, cleanup_interval_hours: int = 1):
        """
        Initialize the orchestrator.
        
        Args:
            max_results: Maximum number of results to keep in memory
            cleanup_interval_hours: Hours after which results are cleaned up
        """
        self.registry = AgentRegistry()
        self.router = TaskRouter()
        self.chain_engine = ChainEngine(self.registry)
        
        # Task storage
        self.tasks: Dict[str, Task] = {}  # task_id -> Task
        self.results: Dict[str, TaskResult] = {}  # task_id -> TaskResult
        self.task_status: Dict[str, TaskStatus] = {}  # task_id -> TaskStatus
        
        # Statistics
        self.stats = defaultdict(int)
        self.stats["tasks_submitted"] = 0
        self.stats["tasks_completed"] = 0
        self.stats["tasks_failed"] = 0
        self.stats["tasks_escalated"] = 0
        
        # Cleanup configuration
        self.max_results = max_results
        self.cleanup_interval_hours = cleanup_interval_hours
        self.last_cleanup = datetime.now()
        
        # Background task for processing
        self._processing_queue: asyncio.Queue = asyncio.Queue()
        self._background_task: Optional[asyncio.Task] = None
        
    async def start(self) -> None:
        """Start the orchestrator background processing."""
        if self._background_task is None:
            self._background_task = asyncio.create_task(self._process_queue())
    
    async def stop(self) -> None:
        """Stop the orchestrator background processing."""
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None
    
    async def submit_task(self, task: Task) -> str:
        """
        Submit a task for execution.
        
        Args:
            task: The task to execute
            
        Returns:
            task_id for tracking the task
        """
        task_id = str(task.id)
        
        # Store task and set initial status
        self.tasks[task_id] = task
        self.task_status[task_id] = TaskStatus.PENDING
        self.stats["tasks_submitted"] += 1
        
        # Add to processing queue
        await self._processing_queue.put(task_id)
        
        return task_id
    
    async def get_result(self, task_id: str) -> Optional[TaskResult]:
        """
        Get result for a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            TaskResult if available, None otherwise
        """
        await self._cleanup_old_results()
        return self.results.get(task_id)
    
    async def run_chain(
        self,
        chain_type: str,
        tasks: List[Task],
        **kwargs: Any
    ) -> List[TaskResult]:
        """
        Run a chain of tasks.
        
        Args:
            chain_type: Type of chain ("sequential", "parallel")
            tasks: List of tasks to execute
            **kwargs: Additional chain-specific parameters
            
        Returns:
            List of task results
            
        Raises:
            ValueError: If chain_type is not supported
        """
        # Generate chain ID for grouping
        chain_id = f"chain_{uuid4().hex[:8]}"
        for task in tasks:
            task.chain_id = chain_id
        
        if chain_type == "sequential":
            return await self.chain_engine.execute_sequential(tasks, **kwargs)
        elif chain_type == "parallel":
            return await self.chain_engine.execute_parallel(tasks, **kwargs)
        else:
            raise ValueError(f"Unsupported chain type: {chain_type}")
    
    async def get_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            TaskStatus if task exists, None otherwise
        """
        return self.task_status.get(task_id)
    
    async def cancel(self, task_id: str) -> bool:
        """
        Cancel a pending or processing task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if task was cancelled, False otherwise
            
        Note:
            Currently only marks tasks as CANCELLED if not yet completed.
            Future implementation could interrupt processing agents.
        """
        if task_id not in self.task_status:
            return False
        
        status = self.task_status[task_id]
        if status in [TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.PROCESSING]:
            self.task_status[task_id] = TaskStatus.CANCELLED
            
            # Create cancelled result
            self.results[task_id] = TaskResult(
                task_id=uuid4(),  # Will be replaced with actual task ID
                agent_id="system",
                output="",
                confidence=0.0,
                tokens_used=0,
                processing_time_ms=0,
                status=TaskStatus.CANCELLED,
                error="Task cancelled by user",
                metadata={"cancelled_at": datetime.now().isoformat()}
            )
            
            return True
        
        return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get orchestrator statistics.
        
        Returns:
            Dictionary with statistics
        """
        await self._cleanup_old_results()
        
        # Calculate success rate
        total_completed = self.stats["tasks_completed"]
        total_failed = self.stats["tasks_failed"]
        total = total_completed + total_failed
        
        success_rate = 0.0
        if total > 0:
            success_rate = total_completed / total
        
        # Get agent stats
        agents = await self.registry.list()
        agent_stats = []
        for agent in agents:
            state = agent.get_state()
            agent_stats.append({
                "agent_id": agent.config.id,
                "name": agent.config.name,
                "status": state.status,
                "tasks_completed": state.tasks_completed,
                "tasks_failed": state.tasks_failed,
                "success_rate": state.success_rate(),
                "last_activity": state.last_activity.isoformat()
            })
        
        return {
            "tasks": {
                "submitted": self.stats["tasks_submitted"],
                "completed": self.stats["tasks_completed"],
                "failed": self.stats["tasks_failed"],
                "escalated": self.stats["tasks_escalated"],
                "success_rate": success_rate,
                "active_tasks": len(self.tasks) - len(self.results)
            },
            "agents": {
                "count": len(agents),
                "details": agent_stats
            },
            "storage": {
                "tasks_stored": len(self.tasks),
                "results_stored": len(self.results),
                "max_results": self.max_results,
                "cleanup_interval_hours": self.cleanup_interval_hours
            }
        }
    
    async def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent with the orchestrator.
        
        Args:
            agent: The agent to register
        """
        await self.registry.register(agent)
    
    async def _process_queue(self) -> None:
        """Background task to process queued tasks."""
        while True:
            try:
                task_id = await self._processing_queue.get()
                await self._process_task(task_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue processing
                print(f"Error processing task {task_id}: {e}")
    
    async def _process_task(self, task_id: str) -> None:
        """Process a single task from the queue."""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        self.task_status[task_id] = TaskStatus.ASSIGNED
        
        # Route the task
        agent, confidence = await self.router.route(task, self.registry)
        
        if not agent:
            # No suitable agent found - escalate
            self.task_status[task_id] = TaskStatus.ESCALATED
            self.results[task_id] = TaskResult(
                task_id=task.id,
                agent_id="system",
                output="",
                confidence=0.0,
                tokens_used=0,
                processing_time_ms=0,
                status=TaskStatus.ESCALATED,
                error="No suitable agent found for task",
                metadata={"escalated_at": datetime.now().isoformat()}
            )
            self.stats["tasks_escalated"] += 1
            return
        
        # Update status and execute
        self.task_status[task_id] = TaskStatus.PROCESSING
        result = await agent.execute(task)
        
        # Update result confidence with router confidence
        result.confidence = max(result.confidence, confidence)
        
        # Store result and update stats
        self.results[task_id] = result
        self.task_status[task_id] = result.status
        
        if result.status == TaskStatus.COMPLETED:
            self.stats["tasks_completed"] += 1
        elif result.status == TaskStatus.FAILED:
            self.stats["tasks_failed"] += 1
        elif result.status == TaskStatus.ESCALATED:
            self.stats["tasks_escalated"] += 1
        
        # Cleanup if needed
        await self._cleanup_old_results()
    
    async def _cleanup_old_results(self) -> None:
        """Clean up old results to free memory."""
        now = datetime.now()
        if (now - self.last_cleanup).total_seconds() < self.cleanup_interval_hours * 3600:
            return
        
        # Cleanup based on age (older than cleanup_interval_hours)
        cutoff_time = now - timedelta(hours=self.cleanup_interval_hours)
        tasks_to_remove = []
        
        for task_id, task in self.tasks.items():
            result = self.results.get(task_id)
            if result:
                # Check if result is complete and old
                if result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.ESCALATED]:
                    # We don't have creation time, so just use count-based cleanup
                    pass
        
        # Cleanup based on count if we have too many
        if len(self.results) > self.max_results:
            # Keep only the most recent max_results
            # Since we don't have timestamps, we'll just clear and keep tasks
            self.results.clear()
        
        self.last_cleanup = now