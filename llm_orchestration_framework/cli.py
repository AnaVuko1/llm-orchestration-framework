"""
Command-line interface for LLM Orchestration Framework.
"""

import asyncio
import sys
from typing import Optional

import click

from .api.server import app
from .core.orchestrator import Orchestrator


@click.group()
@click.version_option()
def cli():
    """LLM Orchestration Framework CLI."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to.")
@click.option("--port", default=8000, help="Port to bind to.")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development.")
def serve(host: str, port: int, reload: bool):
    """Start the API server."""
    import uvicorn
    
    uvicorn.run(
        "llm_orchestration_framework.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


@cli.command()
@click.option("--test-name", help="Run specific test by name.")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
def test(test_name: Optional[str], verbose: bool):
    """Run tests."""
    import subprocess
    
    cmd = ["pytest", "tests/"]
    if verbose:
        cmd.append("-v")
    if test_name:
        cmd.extend(["-k", test_name])
    
    subprocess.run(cmd)


@cli.command()
@click.option("--task-input", required=True, help="Task input text.")
@click.option("--capabilities", help="Comma-separated capabilities.")
@click.option("--priority", default=3, help="Task priority (1-5).")
def submit_task(task_input: str, capabilities: Optional[str], priority: int):
    """Submit a task from the command line."""
    import asyncio
    from .core.task import Task
    
    async def _submit():
        orchestrator = Orchestrator()
        await orchestrator.start()
        
        metadata = {}
        if capabilities:
            metadata["capabilities"] = [c.strip() for c in capabilities.split(",")]
        
        task = Task(
            input=task_input,
            metadata=metadata,
            priority=priority
        )
        
        task_id = await orchestrator.submit_task(task)
        click.echo(f"Task submitted: {task_id}")
        
        # Wait a bit for processing
        await asyncio.sleep(0.5)
        
        result = await orchestrator.get_result(task_id)
        if result:
            click.echo(f"Result: {result.output}")
            click.echo(f"Status: {result.status}")
            click.echo(f"Confidence: {result.confidence:.2f}")
        else:
            click.echo("Task still processing...")
        
        await orchestrator.stop()
    
    asyncio.run(_submit())


@cli.command()
def stats():
    """Show orchestrator statistics."""
    async def _stats():
        orchestrator = Orchestrator()
        await orchestrator.start()
        
        stats = await orchestrator.get_stats()
        
        click.echo("=== Orchestrator Statistics ===")
        click.echo(f"Tasks submitted: {stats['tasks']['submitted']}")
        click.echo(f"Tasks completed: {stats['tasks']['completed']}")
        click.echo(f"Tasks failed: {stats['tasks']['failed']}")
        click.echo(f"Success rate: {stats['tasks']['success_rate']:.1%}")
        click.echo(f"Active tasks: {stats['tasks']['active_tasks']}")
        click.echo(f"Agents registered: {stats['agents']['count']}")
        
        await orchestrator.stop()
    
    asyncio.run(_stats())


@cli.command()
@click.option("--format", type=click.Choice(["text", "json"]), default="text")
def list_agents(format: str):
    """List registered agents."""
    async def _list():
        orchestrator = Orchestrator()
        await orchestrator.start()
        
        agents = await orchestrator.registry.list()
        
        if format == "json":
            import json
            agent_list = []
            for agent in agents:
                state = agent.get_state()
                agent_list.append({
                    "id": agent.config.id,
                    "name": agent.config.name,
                    "capabilities": agent.config.capabilities,
                    "status": state.status,
                    "success_rate": state.success_rate()
                })
            click.echo(json.dumps(agent_list, indent=2))
        else:
            for agent in agents:
                state = agent.get_state()
                click.echo(f"{agent.config.id} ({agent.config.name})")
                click.echo(f"  Capabilities: {', '.join(agent.config.capabilities)}")
                click.echo(f"  Status: {state.status}")
                click.echo(f"  Success rate: {state.success_rate():.1%}")
                click.echo()
        
        await orchestrator.stop()
    
    asyncio.run(_list())


def main():
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()