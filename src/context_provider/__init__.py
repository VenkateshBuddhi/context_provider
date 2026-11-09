"""
Context Provider — Local Evidence-Based Context Handoff System for AI Coding Agents.

Quick start::

    from context_provider import ContextProvider

    async def main():
        async with ContextProvider(workspace_path=".") as client:
            await client.record_note("Starting implementation of auth module", category="goal")
            context = await client.get_active_context()
            artifacts = await client.export()
"""

__version__ = "0.1.0"
__author__ = "Venkatesh Buddhi"
__license__ = "MIT"

from context_provider.sdk import ContextProvider
from context_provider.storage.db import DatabaseManager
from context_provider.storage.models import EventRecord, EventType, EpisodeStatus, EpisodeRecord
from context_provider.export.exporter import ExportEngine
from context_provider.mcp.server import ContextMCPServer

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "ContextProvider",
    "DatabaseManager",
    "ExportEngine",
    "ContextMCPServer",
    "EventRecord",
    "EpisodeRecord",
    "EventType",
    "EpisodeStatus",
]
