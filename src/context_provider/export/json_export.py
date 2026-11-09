"""
JSON Exporter for structured context handoff.

Outputs machine-readable `context.json` adhering to strict schema
for consumption by programmatic agents, MCP servers, and evaluation harnesses.
"""

import json
from typing import List, Dict, Any
from context_provider.storage.models import ProjectRecord, SessionRecord, EpisodeRecord, EventRecord


class JsonExporter:
    """Exports project context into structured JSON format."""

    def render_context_json(
        self,
        project: ProjectRecord,
        session: SessionRecord,
        episodes: List[EpisodeRecord],
        events: List[EventRecord],
        rules_findings: List[Dict[str, Any]],
        contradictions: List[Dict[str, Any]],
    ) -> str:
        """Serializes full context into indented JSON."""
        data = {
            "schema_version": "1.0.0",
            "project": project.to_dict(),
            "session": session.to_dict(),
            "episodes": [ep.to_dict() for ep in episodes],
            "analysis": {
                "traps_avoided": rules_findings,
                "contradictions_detected": contradictions,
            },
            "events_sample": [ev.to_dict() for ev in events[-50:]],
        }
        return json.dumps(data, indent=2)

