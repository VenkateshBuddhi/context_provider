"""
Model Context Protocol (MCP) Server Interface for Context Provider.

Exposes context handoff capabilities as standard MCP tools and resources
usable by AI coding agents (Claude Desktop, Cursor, Antigravity, etc.).
Implements standard JSON-RPC 2.0 over stdio transport.
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from context_provider.config import default_config
from context_provider.storage.db import DatabaseManager
from context_provider.storage.models import EventRecord, EventType, EpisodeStatus
from context_provider.export.exporter import ExportEngine
from context_provider.capture.git_tracker import GitTracker
from context_provider.analysis.detector import ConflictDetector


class ContextMCPServer:
    """Provides programmatic context tool endpoints for AI agents via MCP."""

    def __init__(self, db: DatabaseManager, workspace_root: Path):
        self.db = db
        self.workspace_root = workspace_root.resolve()
        self.export_engine = ExportEngine(db, self.workspace_root)
        self.conflict_detector = ConflictDetector()

    async def get_active_context(self) -> Dict[str, Any]:
        """Tool: Returns high-level active project state, current focus, and developer notes."""
        project = await self.db.get_or_create_project(str(self.workspace_root))
        session = await self.db.get_active_session(project.id)
        if not session:
            return {
                "status": "inactive",
                "message": "No active context session found for this workspace.",
                "workspace": str(self.workspace_root),
            }

        events = await self.db.get_session_events(session.id)
        episodes = await self.db.get_session_episodes(session.id)

        notes = [
            e.payload.get("note") for e in events if e.event_type == EventType.NOTE
        ]

        # Calculate freshness of the session
        latest_ts = events[-1].timestamp if events else session.started_at
        staleness_info = self.conflict_detector.check_staleness(latest_ts)

        # Git status
        git = GitTracker(self.workspace_root, session.id)
        git_info = {
            "branch": git.get_current_branch() if git.is_git_repo() else "N/A",
            "head": git.get_head_commit() if git.is_git_repo() else "N/A",
        }

        # Recent episode summary
        recent_episodes_data = [
            {
                "id": ep.id,
                "title": ep.title,
                "status": ep.status.value,
                "confidence": ep.confidence_score,
                "summary": ep.summary,
            }
            for ep in episodes[-5:]
        ]

        return {
            "status": "active",
            "project": project.name,
            "workspace": str(self.workspace_root),
            "session_id": session.id,
            "session_started_at": session.started_at,
            "freshness": staleness_info,
            "git": git_info,
            "recent_notes": notes[-10:],
            "recent_episodes": recent_episodes_data,
            "total_events_recorded": len(events),
        }

    async def get_handoff_dossier(self) -> Dict[str, Any]:
        """Tool: Returns failed attempts, traps avoided, and empirical evidence."""
        project = await self.db.get_or_create_project(str(self.workspace_root))
        session = await self.db.get_active_session(project.id)
        if not session:
            return {"error": "No active session available for this workspace."}

        # Run export engine to update or produce handoff documents
        files = await self.export_engine.export_all(session.id)
        handoff_content = files["HANDOFF.md"].read_text(encoding="utf-8")
        context_content = files["CONTEXT.md"].read_text(encoding="utf-8")

        # Load JSON representation for programmatic access
        context_json_raw = files["context.json"].read_text(encoding="utf-8")
        context_data = json.loads(context_json_raw)

        return {
            "session_id": session.id,
            "project": project.name,
            "handoff_markdown": handoff_content,
            "context_markdown": context_content,
            "failed_attempts": [
                ep for ep in context_data.get("episodes", [])
                if ep.get("status") == "ABANDONED"
            ],
            "traps_avoided": context_data.get("traps_avoided", []),
            "contradictions": context_data.get("contradictions_detected", []),
        }

    async def record_agent_note(
        self, note: str, category: str = "agent_decision"
    ) -> Dict[str, Any]:
        """Tool: Allows an AI coding agent to record decisions, hypotheses, or observed errors."""
        project = await self.db.get_or_create_project(str(self.workspace_root))
        session = await self.db.get_active_session(project.id)
        if not session:
            session = await self.db.create_session(project.id, metadata={"created_by_agent": True})

        event = EventRecord(
            session_id=session.id,
            event_type=EventType.NOTE,
            payload={"note": note, "category": category},
            evidence_meta={"source": "mcp_agent"},
        )
        await self.db.log_event(event)
        return {
            "success": True,
            "event_id": event.id,
            "session_id": session.id,
            "recorded_note": note,
            "category": category,
        }

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Returns standard MCP tool schemas."""
        return [
            {
                "name": "get_active_context",
                "description": (
                    "Retrieves current active project goals, developer notes, recent episodes, "
                    "git status, and freshness metrics from the local context provider."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "get_handoff_dossier",
                "description": (
                    "Retrieves comprehensive handoff dossier including failed attempts, "
                    "traps to avoid, LAYA contradictions detected, and rendered HANDOFF.md."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "record_agent_note",
                "description": (
                    "Records an agent decision, observation, hypothesis, or error into the "
                    "workspace's local context stream."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "note": {
                            "type": "string",
                            "description": "Content of the note or decision.",
                        },
                        "category": {
                            "type": "string",
                            "description": "Category: goal, decision, bug, observation, or agent_decision.",
                            "default": "agent_decision",
                        },
                    },
                    "required": ["note"],
                },
            },
        ]

    async def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches an MCP tool call."""
        if name == "get_active_context":
            return await self.get_active_context()
        elif name == "get_handoff_dossier":
            return await self.get_handoff_dossier()
        elif name == "record_agent_note":
            note = arguments.get("note", "")
            category = arguments.get("category", "agent_decision")
            return await self.record_agent_note(note, category)
        else:
            raise ValueError(f"Unknown tool: {name}")

    async def run_stdio_server(self) -> None:
        """Runs the MCP server reading JSON-RPC from stdin and writing to stdout."""
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            line = await reader.readline()
            if not line:
                break

            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue

            try:
                request = json.loads(line_str)
                response = await self._handle_jsonrpc(request)
                if response is not None:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
            except Exception as e:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32603, "message": str(e)},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()

    async def _handle_jsonrpc(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handles a single JSON-RPC 2.0 request."""
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                    },
                    "serverInfo": {
                        "name": "context-provider-mcp",
                        "version": "0.1.0",
                    },
                },
            }

        elif method == "notifications/initialized":
            return None

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": self.get_tool_definitions()},
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            try:
                tool_result = await self.handle_tool_call(tool_name, tool_args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": json.dumps(tool_result, indent=2)}
                        ]
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "isError": True,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {str(e)}"}]
                    },
                }

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }


async def main():
    """CLI entrypoint to run the MCP server over stdio."""
    import argparse
    parser = argparse.ArgumentParser(description="Context Provider MCP Server")
    parser.add_argument("-p", "--path", default=".", help="Workspace root directory")
    args = parser.parse_args()

    workspace = Path(args.path).resolve()
    db = DatabaseManager()
    await db.initialize()

    server = ContextMCPServer(db, workspace)
    await server.run_stdio_server()


if __name__ == "__main__":
    asyncio.run(main())
