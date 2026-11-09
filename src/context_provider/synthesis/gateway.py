"""
Phase 3: Synthesis Gateway for Local Generative Engine (llama3.1:latest via Ollama).

Receives compressed, LAYA-filtered episode logs and generates technical markdown
summaries for CONTEXT.md and HANDOFF.md with zero context window choke.
"""

import json
import re
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional
from context_provider.config import default_config
from context_provider.storage.models import EventRecord, EpisodeRecord, EventType
from context_provider.synthesis.prompts import (
    EPISODE_SUMMARY_PROMPT,
    TRAPS_AVOIDED_PROMPT,
    CONTEXT_FOCUS_PROMPT,
)


class SynthesisGateway:
    """Interacts with local Ollama llama3.1 instance to synthesize technical handoff dossiers."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: int = 60,
    ):
        self.base_url = (base_url or default_config.ollama_base_url).rstrip("/")
        self.model = model or default_config.ollama_model
        self.timeout_seconds = timeout_seconds

    def compress_events(self, events: List[EventRecord]) -> str:
        """
        Compresses raw events into high-density textual representations
        to minimize prompt tokens and avoid context choke.
        """
        lines = []
        for ev in events:
            time_part = ev.timestamp[11:19]
            if ev.event_type == EventType.FILE:
                action = ev.payload.get("action", "touch")
                path = ev.payload.get("relative_path") or ev.payload.get("file_path", "")
                size = ev.payload.get("size_bytes", "")
                size_str = f" ({size}B)" if size else ""
                lines.append(f"[{time_part}] FILE {action.upper()}: {path}{size_str}")

            elif ev.event_type == EventType.GIT:
                action = ev.payload.get("action", "vcs")
                subj = ev.payload.get("subject", "")
                is_revert = " [REVERT]" if ev.payload.get("is_revert") else ""
                lines.append(f"[{time_part}] GIT {action.upper()}{is_revert}: {subj}")

            elif ev.event_type in (EventType.COMMAND, EventType.TEST):
                cmd = ev.payload.get("command", "")
                code = ev.payload.get("exit_code", 0)
                # Check stderr_tail, stderr, error; fall back to stdout_tail if non-zero exit and stderr is empty (e.g. pytest, node)
                raw_err = (
                    ev.payload.get("stderr_tail")
                    or ev.payload.get("stderr")
                    or ev.payload.get("error")
                    or ""
                ).strip()
                if not raw_err and code != 0:
                    raw_err = (
                        ev.payload.get("stdout_tail")
                        or ev.payload.get("stdout")
                        or ""
                    ).strip()

                err_snippet = ""
                if code != 0 and raw_err:
                    # In tracebacks/compilation logs, the fatal exception/assertion resides at the bottom
                    err_lines = [line.strip() for line in raw_err.splitlines() if line.strip()]
                    tail_err = " | ".join(err_lines[-4:]) if len(err_lines) > 4 else " | ".join(err_lines)
                    err_snippet = f" | ERROR: {tail_err[:400]}"

                lines.append(f"[{time_part}] CMD (exit {code}): {cmd}{err_snippet}")

            elif ev.event_type == EventType.NOTE:
                cat = ev.payload.get("category", "note").upper()
                note = ev.payload.get("note", "")
                lines.append(f"[{time_part}] NOTE [{cat}]: {note}")

        # Smart cap: Prioritize retaining high-signal events (failures, notes, git) over bulk file edits
        if len(lines) > 25:
            high_signal = [l for l in lines if "exit " in l or "NOTE [" in l or "GIT " in l]
            other = [l for l in lines if l not in high_signal]
            needed = max(0, 25 - len(high_signal))
            lines = sorted(other[-needed:] + high_signal, key=lambda l: l[:10])

        return "\n".join(lines[-25:])

    def sanitize_summary(self, text: str) -> str:
        """
        Post-processing sanitization step to strictly eliminate conversational boilerplate.
        Locates the first occurrence of a markdown bullet point ('- ' or '* ')
        and strips away all text preceding it.
        Fallback: if no bullet points are found, returns the raw text stripped of
        leading/trailing whitespace to prevent losing the summary entirely.
        """
        if not text:
            return ""

        # 1. Prefer bullet point at the start of a line (handles multi-line boilerplate cleanly)
        line_start_match = re.search(r"(?:^|\n)\s*([*-]\s+)", text)
        if line_start_match:
            return text[line_start_match.start(1):].strip()

        # 2. Fallback to any inline bullet point ('- ' or '* ')
        inline_match = re.search(r"[-*]\s+", text)
        if inline_match:
            return text[inline_match.start():].strip()

        # 3. Fallback: return raw text stripped of leading/trailing whitespace
        return text.strip()

    _sanitize_summary = sanitize_summary

    def summarize_episode(self, episode: EpisodeRecord, events: List[EventRecord]) -> str:
        """Generates a concise markdown summary of an episode using llama3.1."""
        compressed = self.compress_events(events)
        prompt = EPISODE_SUMMARY_PROMPT.format(
            title=episode.title,
            status=episode.status.value,
            files=", ".join(episode.files_touched) or "None",
            compressed_events=compressed or "No events",
        )

        llm_response = self._call_ollama(prompt, max_tokens=180)
        if llm_response:
            return self.sanitize_summary(llm_response)

        return self._generate_fallback_summary(episode, events)

    def synthesize_traps_avoided(
        self,
        project_name: str,
        rules_findings: List[Dict[str, Any]],
        contradictions: List[Dict[str, Any]],
        abandoned_episodes: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Synthesizes high-level actionable traps to avoid for HANDOFF.md using llama3.1."""
        if not rules_findings and not contradictions and not abandoned_episodes:
            return None

        rules_str = "\n".join(
            f"- [{r.get('pattern')}] {r.get('description')}" for r in rules_findings[:5]
        ) or "None detected."

        contra_str = "\n".join(
            f"- [LAYA Conflict] Premise '{c.get('premise_note')}' contradicted by: {c.get('invalidating_evidence')}"
            for c in contradictions[:5]
        ) or "None detected."

        abandoned_str = "\n".join(
            f"- {a.get('title')}: {a.get('evidence')}" for a in abandoned_episodes[:5]
        ) or "None."

        prompt = TRAPS_AVOIDED_PROMPT.format(
            project_name=project_name,
            rules_findings=rules_str,
            contradictions=contra_str,
            abandoned_episodes=abandoned_str,
        )

        resp = self._call_ollama(prompt, max_tokens=220)
        return self.sanitize_summary(resp) if resp else None

    def synthesize_active_focus(
        self, notes: List[str], recent_episodes: List[EpisodeRecord]
    ) -> Optional[str]:
        """Synthesizes current active developer focus for CONTEXT.md using llama3.1."""
        if not notes and not recent_episodes:
            return None

        notes_str = "\n".join(f"- {n}" for n in notes[-5:]) or "No recent notes."
        ep_str = "\n".join(
            f"- {ep.title} [{ep.status.value}]" for ep in recent_episodes[-3:]
        ) or "No completed episodes."

        prompt = CONTEXT_FOCUS_PROMPT.format(
            notes=notes_str,
            recent_episodes=ep_str,
        )

        resp = self._call_ollama(prompt, max_tokens=150)
        return self.sanitize_summary(resp) if resp else None

    def _call_ollama(self, prompt: str, max_tokens: int = 180) -> Optional[str]:
        """Calls the local Ollama /api/generate endpoint."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": max_tokens,
                "stop": ["<|eot_id|>", "<|start_header_id|>"],
            },
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                output = data.get("response", "").strip()
                return output if output else None
        except (urllib.error.URLError, TimeoutError, Exception):
            return None

    def _generate_fallback_summary(
        self, episode: EpisodeRecord, events: List[EventRecord]
    ) -> str:
        """Heuristic summary generated when local LLM is temporarily unavailable."""
        notes = [e.payload.get("note") for e in events if e.event_type == EventType.NOTE]
        failed_commands = [
            e
            for e in events
            if e.event_type in (EventType.COMMAND, EventType.TEST)
            and e.payload.get("exit_code") != 0
        ]

        summary_parts = [
            f"- **Episode Status**: {episode.status.value.capitalize()} ({len(events)} events recorded).",
            f"- **Files Involved**: {', '.join(episode.files_touched) if episode.files_touched else 'None'}.",
        ]

        if notes:
            summary_parts.append(f"- **Key Note**: {notes[-1]}")
        if failed_commands:
            last_fail = failed_commands[-1]
            fail_cmd = last_fail.payload.get("command", "")
            raw_err = (last_fail.payload.get("stderr_tail") or last_fail.payload.get("stdout_tail") or "").strip()
            err_line = [l.strip() for l in raw_err.splitlines() if l.strip()][-1] if raw_err else "non-zero return code"
            summary_parts.append(f"- **Failed Execution**: `{fail_cmd}` failed ({err_line[:150]}).")

        return "\n".join(summary_parts)
