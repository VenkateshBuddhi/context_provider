"""
Configuration management for the Local Context Handoff System.

Defines default paths, SQLite storage locations, watcher ignore patterns,
Ollama synthesis parameters, and system thresholds.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os


@dataclass
class Config:
    """System-wide configuration settings."""

    # Default storage directory under user home (~/.local-context)
    base_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("CONTEXT_STORE_DIR", Path.home() / ".local-context")
        )
    )

    # SQLite DB filename
    db_name: str = "context_store.db"

    # Ollama synthesis endpoint and model
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:latest"

    # LAYA BERT Model configuration and local cache paths
    laya_model_id: str = "convaiinnovations/laya"
    laya_local_path: Optional[Path] = field(
        default_factory=lambda: Path(
            r"D:\hf_cache\hub\models--convaiinnovations--laya\snapshots\55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
        ) if Path(r"D:\hf_cache\hub\models--convaiinnovations--laya\snapshots\55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851").exists() else None
    )

    # Capture watcher debounce in seconds
    watcher_debounce_seconds: float = 1.0

    # Episode temporal clustering threshold (idle gap before splitting episode)
    episode_idle_threshold_seconds: int = 300

    # Maximum raw events retained per episode before compaction
    max_events_per_episode: int = 500

    # Default patterns to ignore during file watching
    default_ignore_patterns: List[str] = field(
        default_factory=lambda: [
            ".git/*",
            ".git/**",
            "__pycache__/*",
            "__pycache__/**",
            "*.pyc",
            ".venv/*",
            ".venv/**",
            "node_modules/*",
            "node_modules/**",
            ".DS_Store",
            "*.log",
            "*.tmp",
            "*.swp",
            ".local-context/*",
            ".local-context/**",
        ]
    )

    @property
    def db_path(self) -> Path:
        """Returns the full path to the SQLite database file."""
        return self.base_dir / self.db_name

    def ensure_directories(self) -> None:
        """Ensures that the base storage directory exists on disk."""
        self.base_dir.mkdir(parents=True, exist_ok=True)


# Global default configuration instance
default_config = Config()

