"""
Data models for the Context Provider storage layer.

Defines schemas and serializable entities for:
- Projects (Workspace roots mapped to IDs)
- Sessions (Continuous developer working sessions)
- Episodes (Grouped logical tasks with outcome states)
- Events (Chronological stream: file change, git action, command execution, manual note)
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import json
import uuid


class EventType(str, Enum):
    """Categorization for recorded developer activity events."""
    FILE = "file"
    GIT = "git"
    COMMAND = "command"
    NOTE = "note"
    TEST = "test"


class EpisodeStatus(str, Enum):
    """Lifecycle status assigned to an episode."""
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    INCONCLUSIVE = "inconclusive"


def utc_now_iso() -> str:
    """Returns current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ProjectRecord:
    """Represents a monitored project repository."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    root_path: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionRecord:
    """Represents an active or historical development session."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = ""
    started_at: str = field(default_factory=utc_now_iso)
    ended_at: Optional[str] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeRecord:
    """
    Logical grouping of events representing a specific task attempt.
    Status can be Active, Completed, Abandoned, or Inconclusive.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    title: str = "Untitled Episode"
    status: EpisodeStatus = EpisodeStatus.ACTIVE
    started_at: str = field(default_factory=utc_now_iso)
    ended_at: Optional[str] = None
    confidence_score: float = 0.5
    summary: Optional[str] = None
    files_touched: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, EpisodeStatus) else self.status
        return d


@dataclass
class EventRecord:
    """
    Raw immutable event in the chronological transaction log.
    Payload contains specific details for file diffs, git hashes, terminal output, etc.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    episode_id: Optional[str] = None
    event_type: EventType = EventType.NOTE
    timestamp: str = field(default_factory=utc_now_iso)
    payload: Dict[str, Any] = field(default_factory=dict)
    evidence_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value if isinstance(self.event_type, EventType) else self.event_type
        return d

