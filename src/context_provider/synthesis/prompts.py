"""
Prompt Templates for Local Generative Synthesis (llama3.1:latest).

Carefully engineered to yield concise, structured, factual handoff summaries
without hallucinations, pleasantries, or context window saturation.
"""

EPISODE_SUMMARY_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert context synthesizer for autonomous AI coding agents.
Summarize the following developer episode into 2-3 concise markdown bullet points.
Strictly ground your answer on the provided facts. Do not invent details.
Output ONLY the markdown bullet points. Do not include any introductory phrases, conversational filler, or acknowledging text.
Format:
- **Attempt**: Brief technical description of what was attempted.
- **Evidence/Outcome**: Exact technical result. When a command or test fails, you MUST explicitly name the specific error or exception message from the log (e.g. ReferenceError, SyntaxError, AssertionError, compiler error). NEVER summarize as simply "exit code 1 error" or "command failed"—always quote or state the specific failure reason.
- **Status**: Current technical outcome.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Episode Title: {title}
Status: {status}
Files Touched: {files}
Raw Events Log:
{compressed_events}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

TRAPS_AVOIDED_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are preparing a critical "Traps to Avoid" dossier for an incoming AI coding agent.
Review the following detected failures, command thrashing, and abandoned attempts.
Synthesize 1-3 crisp warnings advising the next agent on what NOT to repeat and WHY.
Output ONLY the markdown bullet points. Do not include any introductory phrases, conversational filler, or acknowledging text.
Always name the specific error, exception, or failure reason (e.g. ReferenceError, missing dependencies, compiler errors) so the incoming agent knows exactly what went wrong.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Project: {project_name}
Detected Anomalies & Rules Engine Findings:
{rules_findings}

Contradictions Identified by LAYA:
{contradictions}

Abandoned Episodes:
{abandoned_episodes}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

CONTEXT_FOCUS_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI context synthesizer. Summarize the developer's current focus, active decisions,
and immediate next steps for CONTEXT.md based only on the provided session notes and recent episodes.
Provide 2-4 concise markdown bullet points.
Output ONLY the markdown bullet points. Do not include any introductory phrases, conversational filler, or acknowledging text.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Recent Notes:
{notes}

Recent Completed Work:
{recent_episodes}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
