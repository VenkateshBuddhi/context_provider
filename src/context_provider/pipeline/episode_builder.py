"""
Phase 2: Advanced Episode Builder.

Groups chronological raw developer events into cohesive logical episodes using:
1. Temporal proximity clustering (idle gap threshold).
2. Hard boundary triggers (Git commits, git reverts, branch switches, goal notes).
3. File overlap metrics (Jaccard similarity on touched files to detect subsystem switches).
4. Deterministic state classification: [Active, Completed, Abandoned, Inconclusive].
"""

from datetime import datetime, timezone
from typing import List, Set, Dict, Any, Optional, Tuple
from context_provider.storage.models import EventRecord, EpisodeRecord, EpisodeStatus, EventType
from context_provider.config import default_config


class EpisodeCluster:
    """Internal cluster representation during segmentation."""

    def __init__(self, initial_event: EventRecord, boundary_reason: str = "SESSION_START"):
        self.events: List[EventRecord] = [initial_event]
        self.boundary_reason: str = boundary_reason
        self.files_touched: Set[str] = set()
        self._add_files(initial_event)

    def _extract_files(self, event: EventRecord) -> Set[str]:
        files = set()
        if event.event_type == EventType.FILE:
            rel = event.payload.get("relative_path") or event.payload.get("file_path")
            if rel:
                files.add(rel.replace("\\", "/"))
            if "src_rel" in event.payload:
                files.add(event.payload["src_rel"].replace("\\", "/"))
            if "dest_rel" in event.payload:
                files.add(event.payload["dest_rel"].replace("\\", "/"))
        elif event.event_type == EventType.GIT:
            diff_summary = event.payload.get("diff_summary", "")
            for line in diff_summary.splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    files.add(parts[-1].replace("\\", "/"))
        return files

    def _add_files(self, event: EventRecord) -> None:
        self.files_touched.update(self._extract_files(event))

    def add_event(self, event: EventRecord) -> None:
        self.events.append(event)
        self._add_files(event)

    @property
    def start_time(self) -> float:
        return datetime.fromisoformat(self.events[0].timestamp.replace("Z", "+00:00")).timestamp()

    @property
    def end_time(self) -> float:
        return datetime.fromisoformat(self.events[-1].timestamp.replace("Z", "+00:00")).timestamp()

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_time - self.start_time)


class EpisodeBuilder:
    """Clusters raw events into task episodes and determines task resolution status."""

    def __init__(
        self,
        idle_gap_seconds: Optional[int] = None,
        file_overlap_threshold: float = 0.15,
        subsystem_pause_seconds: float = 45.0,
        min_events_threshold: int = 3,
    ):
        self.idle_gap_seconds = idle_gap_seconds or default_config.episode_idle_threshold_seconds
        self.file_overlap_threshold = file_overlap_threshold
        self.subsystem_pause_seconds = subsystem_pause_seconds
        self.min_events_threshold = min_events_threshold
        self.active_event_buffer: List[EventRecord] = []

    @property
    def active_buffer(self) -> List[EventRecord]:
        """Alias for active_event_buffer."""
        return self.active_event_buffer

    @active_buffer.setter
    def active_buffer(self, value: List[EventRecord]) -> None:
        self.active_event_buffer = value

    def _parse_timestamp(self, ts_str: str) -> float:
        """Parses ISO timestamp to UNIX epoch seconds."""
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()

    def is_high_signal_event(self, event: EventRecord) -> bool:
        """
        Determines if an event represents a high-signal action:
        - Developer note (EventType.NOTE).
        - Terminal command or test run with non-zero exit code indicating failure.
        - Git commit or revert action.
        """
        # 1. Developer note
        if event.event_type == EventType.NOTE or event.event_type == "note":
            return True

        # 2. Terminal command / test failure (non-zero exit code)
        if event.event_type in (EventType.COMMAND, EventType.TEST, "command", "test"):
            exit_code = event.payload.get("exit_code")
            if exit_code is not None:
                try:
                    if int(exit_code) != 0:
                        return True
                except (ValueError, TypeError):
                    pass
            if event.payload.get("failed") is True:
                return True

        # 3. Git commit or revert
        if event.event_type in (EventType.GIT, "git"):
            action = str(event.payload.get("action", "")).lower()
            if action in ("commit", "revert"):
                return True
            if event.payload.get("is_commit") is True or event.payload.get("is_revert") is True:
                return True

        return False

    def is_cluster_eligible(self, cluster: EpisodeCluster) -> bool:
        """
        Validates whether an event cluster meets the minimum threshold for episode creation:
        - Contains 3 or more distinct events, OR
        - Contains at least one high-signal event (developer note, command failure, git commit).
        """
        if not cluster or not cluster.events:
            return False
        distinct_events = len({e.id for e in cluster.events})
        if distinct_events >= self.min_events_threshold:
            return True
        return any(self.is_high_signal_event(e) for e in cluster.events)

    def compute_jaccard_similarity(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Calculates Jaccard similarity index between two sets of file paths."""
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def _is_hard_boundary_starter(self, event: EventRecord) -> Tuple[bool, Optional[str]]:
        """Checks if an incoming event represents an explicit new goal or branch switch."""
        if event.event_type == EventType.NOTE:
            category = event.payload.get("category", "").lower()
            note_text = event.payload.get("note", "").lower()
            if category == "goal" or any(kw in note_text for kw in ["goal:", "target:", "objective:", "starting:"]):
                return True, "DEVELOPER_GOAL_NOTE"

        if event.event_type == EventType.GIT and event.payload.get("action") == "branch_switch":
            return True, "GIT_BRANCH_SWITCH"

        return False, None

    def _is_hard_boundary_ender(self, event: EventRecord) -> Tuple[bool, Optional[str]]:
        """Checks if an event checkpoint concludes an episode (e.g. commit or revert)."""
        if event.event_type == EventType.GIT:
            if event.payload.get("is_revert"):
                return True, "GIT_REVERT_CHECKPOINT"
            if event.payload.get("action") == "commit":
                return True, "GIT_COMMIT_CHECKPOINT"

        return False, None

    def cluster_events(self, events: List[EventRecord]) -> List[EpisodeCluster]:
        """
        Segments an event stream into discrete clusters based on:
        1. Temporal idle gaps.
        2. Hard checkpoint triggers (commits, reverts, goal notes).
        3. File overlap disjointness (subsystem context switching).
        """
        if not events:
            return []

        # Sort chronologically
        sorted_events = sorted(events, key=lambda e: self._parse_timestamp(e.timestamp))

        clusters: List[EpisodeCluster] = []
        current = EpisodeCluster(sorted_events[0], boundary_reason="SESSION_START")

        for event in sorted_events[1:]:
            if current is None:
                current = EpisodeCluster(event, boundary_reason="POST_CHECKPOINT")
                # Check if this initial event is itself an ender
                is_ender, ender_reason = self._is_hard_boundary_ender(event)
                if is_ender:
                    current.boundary_reason = ender_reason or "CHECKPOINT"
                    clusters.append(current)
                    current = None
                continue

            prev_event = current.events[-1]
            time_gap = self._parse_timestamp(event.timestamp) - self._parse_timestamp(prev_event.timestamp)

            # Check 1: Temporal Idle Gap
            if time_gap > self.idle_gap_seconds:
                if self.is_cluster_eligible(current):
                    clusters.append(current)
                    current = EpisodeCluster(event, boundary_reason=f"TEMPORAL_IDLE_GAP ({int(time_gap)}s)")
                else:
                    # Minimum threshold not met: retain events in active buffer
                    current.add_event(event)
                continue

            # Check 2: Explicit Goal Starter
            is_goal, goal_reason = self._is_hard_boundary_starter(event)
            if is_goal:
                if self.is_cluster_eligible(current):
                    clusters.append(current)
                    current = EpisodeCluster(event, boundary_reason=goal_reason or "NEW_GOAL")
                else:
                    current.add_event(event)
                continue

            # Check 3: File Overlap & Subsystem Shift
            ev_files = current._extract_files(event)
            if ev_files and current.files_touched and time_gap >= self.subsystem_pause_seconds:
                sim = self.compute_jaccard_similarity(current.files_touched, ev_files)
                if sim < self.file_overlap_threshold:
                    if self.is_cluster_eligible(current):
                        clusters.append(current)
                        current = EpisodeCluster(
                            event,
                            boundary_reason=f"MODULE_SWITCH (Jaccard {sim:.2f}, gap {int(time_gap)}s)",
                        )
                    else:
                        current.add_event(event)
                    continue

            # Append event to current cluster
            current.add_event(event)

            # Check 4: Checkpoint Ender (Git commit / revert concludes this episode)
            is_ender, ender_reason = self._is_hard_boundary_ender(event)
            if is_ender:
                current.boundary_reason = ender_reason or "CHECKPOINT"
                if self.is_cluster_eligible(current):
                    clusters.append(current)
                    current = None

            if current is None and event != sorted_events[-1]:
                # Next event will start a new cluster
                continue

        if current and current.events:
            clusters.append(current)

        return clusters

    def infer_episode_status(
        self,
        cluster: EpisodeCluster,
        is_latest_in_session: bool = False,
        session_is_active: bool = True,
    ) -> EpisodeStatus:
        """
        Assigns an evidence-backed status to the episode:
        - ACTIVE: Ongoing task in an active session within idle limits.
        - COMPLETED: Ends with clean Git commit or green test suite without subsequent failure.
        - ABANDONED: Contains a Git revert, consecutive test failures followed by pivot, or failed bug note.
        - INCONCLUSIVE: Interrupted work, syntax errors, or uncommitted failures.
        """
        events = cluster.events
        if not events:
            return EpisodeStatus.INCONCLUSIVE

        now_ts = datetime.now(timezone.utc).timestamp()
        is_fresh = (now_ts - cluster.end_time) < self.idle_gap_seconds

        # Check for explicit reverts
        for ev in events:
            if ev.event_type == EventType.GIT and ev.payload.get("is_revert"):
                return EpisodeStatus.ABANDONED

        # Check for clean commit
        last_git = next((e for e in reversed(events) if e.event_type == EventType.GIT), None)
        if last_git and last_git.payload.get("action") == "commit" and not last_git.payload.get("is_revert"):
            return EpisodeStatus.COMPLETED

        # Check for test outcomes
        test_events = [e for e in events if e.event_type == EventType.TEST]
        if test_events and test_events[-1].payload.get("exit_code") == 0:
            return EpisodeStatus.COMPLETED

        if is_latest_in_session and session_is_active and is_fresh:
            return EpisodeStatus.ACTIVE

        # Check notes for abandoned indicators
        notes = [e for e in events if e.event_type == EventType.NOTE]
        for n in notes:
            n_text = n.payload.get("note", "").lower()
            if any(kw in n_text for kw in ["abandoned", "reverted", "scrapped", "trap:"]):
                return EpisodeStatus.ABANDONED

        return EpisodeStatus.INCONCLUSIVE

    def generate_episode_title(self, cluster: EpisodeCluster) -> str:
        """Synthesizes a descriptive title from goals, commits, notes, or files."""
        # Check for explicit goal note
        for ev in cluster.events:
            if ev.event_type == EventType.NOTE:
                cat = ev.payload.get("category", "")
                note = ev.payload.get("note", "")
                if cat in ("goal", "decision"):
                    return f"{cat.capitalize()}: {note[:60]}"

        # Check for commit
        for ev in reversed(cluster.events):
            if ev.event_type == EventType.GIT and ev.payload.get("action") == "commit":
                return f"Commit: {ev.payload.get('subject', 'VCS Commit')[:60]}"

        # Check for test execution
        for ev in reversed(cluster.events):
            if ev.event_type == EventType.TEST:
                status_str = "Passed" if ev.payload.get("exit_code") == 0 else "Failed"
                return f"Test Run ({status_str}): {ev.payload.get('command', 'pytest')[:50]}"

        # Check touched files
        if cluster.files_touched:
            files_list = sorted(list(cluster.files_touched))
            if len(files_list) == 1:
                return f"Work on {files_list[0]}"
            return f"Work on {files_list[0]} and {len(files_list) - 1} other files"

        return f"Developer Activity ({len(cluster.events)} events)"

    def build_episodes_from_events(
        self,
        session_id: str,
        events: List[EventRecord],
        session_is_active: bool = True,
    ) -> List[EpisodeRecord]:
        """
        Processes event stream and creates fully structured EpisodeRecord instances
        ready for persistence and agent handoff.

        Enforces minimum threshold validation: a cluster of events is only finalized
        into an Episode object if:
        1. It contains 3 or more distinct events, OR
        2. It contains at least one high-signal event (developer note, command failure, git commit).

        If the cluster fails validation, it is NOT finalized into an Episode; its events
        are retained in self.active_event_buffer so they merge into the next logical episode.
        """
        # Combine lingering active buffer events with incoming events (deduplicate by event id)
        all_events: List[EventRecord] = []
        seen_ids = set()
        for ev in self.active_event_buffer + events:
            if ev.id not in seen_ids:
                seen_ids.add(ev.id)
                all_events.append(ev)

        if not all_events:
            self.active_event_buffer = []
            return []

        clusters = self.cluster_events(all_events)
        episodes: List[EpisodeRecord] = []
        self.active_event_buffer = []

        for i, cluster in enumerate(clusters):
            # Finalization validation check:
            if not self.is_cluster_eligible(cluster):
                # Do not finalize: keep in active event buffer so they merge into next episode
                for ev in cluster.events:
                    if ev.id not in {e.id for e in self.active_event_buffer}:
                        self.active_event_buffer.append(ev)
                continue

            is_latest = (i == len(clusters) - 1)
            status = self.infer_episode_status(
                cluster, is_latest_in_session=is_latest, session_is_active=session_is_active
            )
            title = self.generate_episode_title(cluster)
            started_at = cluster.events[0].timestamp
            ended_at = cluster.events[-1].timestamp

            ep = EpisodeRecord(
                session_id=session_id,
                title=title,
                status=status,
                started_at=started_at,
                ended_at=ended_at,
                files_touched=sorted(list(cluster.files_touched)),
                metadata={
                    "event_count": len(cluster.events),
                    "boundary_reason": cluster.boundary_reason,
                    "duration_seconds": round(cluster.duration_seconds, 2),
                    "first_event_id": cluster.events[0].id,
                    "last_event_id": cluster.events[-1].id,
                    "event_ids": [e.id for e in cluster.events],
                },
            )
            episodes.append(ep)

        return episodes
