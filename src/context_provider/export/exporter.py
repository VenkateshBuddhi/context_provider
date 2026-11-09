"""
Master Export Engine coordinating Phase 3 models (LAYA BERT + llama3.1) and disk generation.

Orchestrates:
1. Event retrieval from SQLite
2. Episode clustering and tagging via EpisodeBuilder
3. Anomaly and failure pattern detection via RulesEngine
4. Schema-safe contradiction and staleness evaluation via LAYA Engine (ConflictDetector)
5. Local LLM generative synthesis via SynthesisGateway (llama3.1:latest)
6. File generation for true model-synthesized CONTEXT.md, HANDOFF.md, and context.json
"""

from pathlib import Path
from typing import Dict, Optional, List
from context_provider.storage.db import DatabaseManager
from context_provider.storage.models import EventRecord, EpisodeStatus, EventType
from context_provider.pipeline.episode_builder import EpisodeBuilder
from context_provider.pipeline.rules_engine import RulesEngine
from context_provider.analysis.detector import ConflictDetector
from context_provider.analysis.confidence import ConfidenceScorer
from context_provider.synthesis.gateway import SynthesisGateway
from context_provider.capture.git_tracker import GitTracker
from context_provider.export.markdown import MarkdownExporter
from context_provider.export.json_export import JsonExporter


class ExportEngine:
    """Coordinates generation of all model-synthesized agent-facing handoff artifacts."""

    def __init__(self, db: DatabaseManager, workspace_root: Path):
        self.db = db
        self.workspace_root = workspace_root.resolve()
        self.episode_builder = EpisodeBuilder()
        self.rules_engine = RulesEngine()
        self.conflict_detector = ConflictDetector()
        self.confidence_scorer = ConfidenceScorer()
        self.synthesis_gateway = SynthesisGateway()
        self.md_exporter = MarkdownExporter()
        self.json_exporter = JsonExporter()

    async def export_all(self, session_id: str) -> Dict[str, Path]:
        """Processes session data and writes CONTEXT.md, HANDOFF.md, and context.json."""
        project = await self.db.get_or_create_project(str(self.workspace_root))
        active_session = await self.db.get_active_session(project.id)
        events = await self.db.get_session_events(session_id)

        # 1. Cluster events into episodes
        episodes = self.episode_builder.build_episodes_from_events(session_id, events)

        # 2. Score confidence & synthesize episode summaries using local LLM
        for ep in episodes:
            ep_events = [
                e for e in events
                if ep.started_at <= e.timestamp <= (ep.ended_at or "9999")
            ]
            ep.confidence_score = self.confidence_scorer.compute_episode_confidence(ep_events, ep.status)
            ep.summary = self.synthesis_gateway.summarize_episode(ep, ep_events)

        # Persist clustered episodes and link events in SQLite
        await self.db.save_episodes(episodes)
        for ep in episodes:
            event_ids = ep.metadata.get("event_ids", [])
            if event_ids:
                await self.db.link_events_to_episode(ep.id, event_ids)

        # 3. Deterministic rules analysis (mechanical patterns: thrashing, reverts, regressions)
        rules_findings = self.rules_engine.analyze_patterns(events)

        # 4. LAYA BERT conflict & contradiction detection
        notes = [e for e in events if e.event_type == EventType.NOTE]
        contradictions = self.conflict_detector.detect_contradictions(notes, events)

        # 5. Git status overview
        git = GitTracker(self.workspace_root, session_id)
        latest_git = {
            "branch": git.get_current_branch() if git.is_git_repo() else "N/A",
            "head": git.get_head_commit() if git.is_git_repo() else "N/A",
        }

        # 6. Extract failed attempts
        failed_attempts = [
            {
                "title": ep.title,
                "status": ep.status.value,
                "files": ep.files_touched,
                "evidence": ep.summary or "Approach was abandoned or reverted.",
            }
            for ep in episodes
            if ep.status == EpisodeStatus.ABANDONED
        ]

        active_notes = [n.payload.get("note", "") for n in notes]

        # 7. Model synthesis: Generative synthesis of active focus & traps avoided
        synthesized_focus = self.synthesis_gateway.synthesize_active_focus(
            notes=active_notes, recent_episodes=episodes
        )
        synthesized_traps = self.synthesis_gateway.synthesize_traps_avoided(
            project_name=project.name,
            rules_findings=rules_findings,
            contradictions=contradictions,
            abandoned_episodes=failed_attempts,
        )

        # 8. Filter episodes for markdown rendering: eliminate low-signal noise
        # Only retain episodes with status Active, Abandoned, or Completed, OR confidence >= 0.5
        # Silently drops INCONCLUSIVE episodes with confidence < 0.5 from both CONTEXT.md and HANDOFF.md
        filtered_episodes = [
            ep for ep in episodes
            if ep.status in (EpisodeStatus.ACTIVE, EpisodeStatus.ABANDONED, EpisodeStatus.COMPLETED)
            or (isinstance(ep.status, str) and ep.status.lower() in ("active", "abandoned", "completed"))
            or ep.confidence_score >= 0.5
        ]

        context_md = self.md_exporter.render_context_md(
            project=project,
            session=active_session,
            episodes=filtered_episodes,
            latest_git=latest_git,
            active_notes=active_notes,
            synthesized_focus=synthesized_focus,
        )

        handoff_md = self.md_exporter.render_handoff_md(
            project=project,
            session=active_session,
            episodes=filtered_episodes,
            failed_attempts=failed_attempts,
            traps_avoided=rules_findings,
            contradictions=contradictions,
            synthesized_traps=synthesized_traps,
        )

        context_json = self.json_exporter.render_context_json(
            project=project,
            session=active_session,
            episodes=episodes,
            events=events,
            rules_findings=rules_findings,
            contradictions=contradictions,
        )

        # 9. Write to workspace disk
        context_md_path = self.workspace_root / "CONTEXT.md"
        handoff_md_path = self.workspace_root / "HANDOFF.md"
        context_json_path = self.workspace_root / "context.json"

        context_md_path.write_text(context_md, encoding="utf-8")
        handoff_md_path.write_text(handoff_md, encoding="utf-8")
        context_json_path.write_text(context_json, encoding="utf-8")

        return {
            "CONTEXT.md": context_md_path,
            "HANDOFF.md": handoff_md_path,
            "context.json": context_json_path,
        }
