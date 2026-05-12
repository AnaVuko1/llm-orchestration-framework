"""
FastAPI server for LLM orchestration framework.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..core.agent import AgentConfig, BaseAgent
from ..core.orchestrator import Orchestrator
from ..core.task import Task
from .schemas import (
    AgentConfigCreate,
    AgentResponse,
    ChainResponse,
    ErrorResponse,
    HealthResponse,
    ParallelChainRequest,
    SequentialChainRequest,
    StatsResponse,
    TaskCreate,
    TaskResponse,
    TaskResultResponse,
)

# Global orchestrator instance
_orchestrator: Orchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    # Startup
    global _orchestrator
    _orchestrator = Orchestrator()
    await _orchestrator.start()
    
    yield
    
    # Shutdown
    if _orchestrator:
        await _orchestrator.stop()
        _orchestrator = None


# Create FastAPI app
app = FastAPI(
    title="LLM Orchestration Framework",
    description="General-purpose LLM orchestration layer — define, chain, route, and escalate agents",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper functions
def get_orchestrator() -> Orchestrator:
    """Get the global orchestrator instance."""
    if _orchestrator is None:
        raise RuntimeError("Orchestrator not initialized")
    return _orchestrator


async def create_agent_from_config(config: AgentConfigCreate) -> BaseAgent:
    """
    Create an agent from configuration.
    
    This is a placeholder - in a real implementation, this would create
    the appropriate agent with the specified provider.
    """
    # Convert to internal config
    internal_config = AgentConfig(**config.model_dump())
    
    # Create agent (this is simplified - would need provider instantiation)
    from ..core.agent import BaseAgent as InternalBaseAgent
    
    class SimpleAgent(InternalBaseAgent):
        async def _execute_internal(self, task):
            # Simple mock implementation
            from ..core.task import TaskResult, TaskStatus
            return TaskResult(
                task_id=task.id,
                agent_id=self.config.id,
                output=f"Mock response for: {task.input[:50]}...",
                confidence=0.9,
                tokens_used=100,
                processing_time_ms=500,
                status=TaskStatus.COMPLETED,
                metadata={"mock": True}
            )
    
    return SimpleAgent(internal_config)


# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    orchestrator = get_orchestrator()
    stats = await orchestrator.get_stats()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="0.1.0",
        agents_registered=stats["agents"]["count"],
        tasks_queued=stats["tasks"]["active_tasks"]
    )


@app.get("/.well-known/ai-agent.json")
async def ai_agent_manifest():
    """AI agent manifest for discoverability."""
    return {
        "name": "LLM Orchestration Framework",
        "description": "General-purpose LLM orchestration layer",
        "version": "0.1.0",
        "api_version": "v1",
        "endpoints": {
            "tasks": "/v1/tasks",
            "agents": "/v1/agents",
            "chains": "/v1/chains",
            "health": "/health"
        },
        "capabilities": [
            "task_orchestration",
            "agent_routing",
            "chain_execution",
            "memory_management"
        ]
    }


# Task endpoints
@app.post("/v1/tasks", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_task(task_data: TaskCreate):
    """Submit a task for execution."""
    try:
        orchestrator = get_orchestrator()
        
        # Create task
        task = Task(
            input=task_data.input,
            agent_id=task_data.agent_id,
            context=task_data.context,
            metadata=task_data.metadata,
            parent_task_id=task_data.parent_task_id,
            chain_id=task_data.chain_id,
            priority=task_data.priority,
            max_retries=task_data.max_retries,
            timeout_seconds=task_data.timeout_seconds
        )
        
        # Submit task
        task_id = await orchestrator.submit_task(task)
        
        return TaskResponse(
            task_id=task.id,
            status=await orchestrator.get_status(task_id),
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/v1/tasks/{task_id}", response_model=TaskResultResponse)
async def get_task_result(task_id: str):
    """Get result for a task."""
    try:
        orchestrator = get_orchestrator()
        result = await orchestrator.get_result(task_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found or still processing"
            )
        
        return TaskResultResponse(**result.model_dump())
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Agent endpoints
@app.post("/v1/agents/register", status_code=status.HTTP_201_CREATED)
async def register_agent(config: AgentConfigCreate):
    """Register a new agent."""
    try:
        orchestrator = get_orchestrator()
        
        # Create and register agent
        agent = await create_agent_from_config(config)
        await orchestrator.register_agent(agent)
        
        return {"message": f"Agent {config.id} registered successfully"}
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/v1/agents", response_model=List[AgentResponse])
async def list_agents():
    """List all registered agents."""
    try:
        orchestrator = get_orchestrator()
        agents = await orchestrator.registry.list()
        
        response = []
        for agent in agents:
            state = agent.get_state()
            response.append(AgentResponse(
                config=AgentConfigCreate(**agent.config.model_dump()),
                state={
                    "agent_id": state.agent_id,
                    "status": state.status,
                    "current_task_id": state.current_task_id,
                    "tasks_completed": state.tasks_completed,
                    "tasks_failed": state.tasks_failed,
                    "last_activity": state.last_activity.isoformat(),
                    "success_rate": state.success_rate()
                }
            ))
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/v1/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str):
    """Get details for a specific agent."""
    try:
        orchestrator = get_orchestrator()
        agent = await orchestrator.registry.get(agent_id)
        
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        state = agent.get_state()
        return AgentResponse(
            config=AgentConfigCreate(**agent.config.model_dump()),
            state={
                "agent_id": state.agent_id,
                "status": state.status,
                "current_task_id": state.current_task_id,
                "tasks_completed": state.tasks_completed,
                "tasks_failed": state.tasks_failed,
                "last_activity": state.last_activity.isoformat(),
                "success_rate": state.success_rate()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Chain endpoints
@app.post("/v1/chains/sequential", response_model=ChainResponse)
async def run_sequential_chain(chain_request: SequentialChainRequest):
    """Run a sequential chain of tasks."""
    try:
        orchestrator = get_orchestrator()
        
        # Convert chain tasks to internal tasks
        tasks = []
        for task_data in chain_request.tasks:
            task = Task(
                input=task_data.input,
                context=task_data.context,
                metadata=task_data.metadata,
                priority=task_data.priority,
                max_retries=task_data.max_retries,
                timeout_seconds=task_data.timeout_seconds
            )
            tasks.append(task)
        
        # Execute chain
        results = await orchestrator.run_chain(
            chain_type="sequential",
            tasks=tasks,
            context_accumulation=chain_request.context_accumulation
        )
        
        # Convert results
        result_responses = [
            TaskResultResponse(**result.model_dump())
            for result in results
        ]
        
        return ChainResponse(
            chain_id=f"seq_{uuid.uuid4().hex[:8]}",
            results=result_responses
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/v1/chains/parallel", response_model=ChainResponse)
async def run_parallel_chain(chain_request: ParallelChainRequest):
    """Run a parallel chain of tasks."""
    try:
        orchestrator = get_orchestrator()
        
        # Convert chain tasks to internal tasks
        tasks = []
        for task_data in chain_request.tasks:
            task = Task(
                input=task_data.input,
                context=task_data.context,
                metadata=task_data.metadata,
                priority=task_data.priority,
                max_retries=task_data.max_retries,
                timeout_seconds=task_data.timeout_seconds
            )
            tasks.append(task)
        
        # Execute chain
        results = await orchestrator.run_chain(
            chain_type="parallel",
            tasks=tasks,
            max_concurrent=chain_request.max_concurrent
        )
        
        # Convert results
        result_responses = [
            TaskResultResponse(**result.model_dump())
            for result in results
        ]
        
        return ChainResponse(
            chain_id=f"par_{uuid.uuid4().hex[:8]}",
            results=result_responses
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Stats endpoint
@app.get("/v1/stats", response_model=StatsResponse)
async def get_stats():
    """Get orchestrator statistics."""
    try:
        orchestrator = get_orchestrator()
        stats = await orchestrator.get_stats()
        
        # Convert agent stats
        agent_details = []
        for agent_stat in stats["agents"]["details"]:
            agent_details.append({
                "agent_id": agent_stat["agent_id"],
                "status": agent_stat["status"],
                "current_task_id": agent_stat["current_task_id"],
                "tasks_completed": agent_stat["tasks_completed"],
                "tasks_failed": agent_stat["tasks_failed"],
                "last_activity": agent_stat["last_activity"],
                "success_rate": agent_stat["success_rate"]
            })
        
        return StatsResponse(
            tasks={
                "submitted": stats["tasks"]["submitted"],
                "completed": stats["tasks"]["completed"],
                "failed": stats["tasks"]["failed"],
                "escalated": stats["tasks"]["escalated"],
                "success_rate": stats["tasks"]["success_rate"],
                "active_tasks": stats["tasks"]["active_tasks"]
            },
            agents={
                "count": stats["agents"]["count"],
                "details": agent_details
            },
            storage={
                "tasks_stored": stats["storage"]["tasks_stored"],
                "results_stored": stats["storage"]["results_stored"],
                "max_results": stats["storage"]["max_results"],
                "cleanup_interval_hours": stats["storage"]["cleanup_interval_hours"]
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            code="HTTP_ERROR"
        ).model_dump()
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle generic exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            code="INTERNAL_ERROR"
        ).model_dump()
    )