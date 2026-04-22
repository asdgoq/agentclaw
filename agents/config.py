#!/usr/bin/env python3
"""Global configuration, constants, and provider initialization."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Try CWD first, then parent directory (for running from agents/ subdir)
    loaded = load_dotenv(override=True)
    if not loaded:
        _parent_env = Path(__file__).resolve().parent.parent / ".env"
        if _parent_env.exists():
            load_dotenv(_parent_env, override=True)
except ImportError:
    pass  # dotenv optional

# Provider selection: "anthropic" or "glm"
PROVIDER = os.getenv("PROVIDER", "glm")

client = None
MODEL = os.getenv("MODEL_ID", "glm-5")

def _init_provider():
    """Lazy-initialize the LLM client. Call before using client/MODEL."""
    global client, MODEL
    if client is not None:
        return

    # Last-resort: try loading .env if key is missing
    if not os.getenv("GLM_API_KEY") and not os.getenv("ANTHROPIC_BASE_URL"):
        try:
            from dotenv import load_dotenv
            # Try multiple locations for .env
            load_dotenv(override=True)
            _paths_to_try = [
                Path.cwd() / ".env",
                Path(__file__).resolve().parent.parent / ".env",
                Path(__file__).resolve().parent / ".env",
            ]
            for _p in _paths_to_try:
                if _p.exists():
                    load_dotenv(_p, override=True)
                    break
        except ImportError:
            pass

    if PROVIDER == "glm":
        try:
            from zai import ZhipuAiClient
            _key = os.getenv("GLM_API_KEY", "")
            if not _key or _key == "dummy" or _key == "your-key-here":
                print(f"[config] WARNING: GLM_API_KEY is not set or is placeholder (got: {_key[:8]}...)", file=__import__("sys").stderr)
                return
            client = ZhipuAiClient(api_key=_key)
        except Exception as e:
            print(f"[config] Failed to init GLM client: {e}", file=__import__("sys").stderr)
    else:
        try:
            from anthropic import Anthropic
            if os.getenv("ANTHROPIC_BASE_URL"):
                os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
            client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
            MODEL = os.environ.get("MODEL_ID", MODEL)
        except Exception as e:
            print(f"[config] Failed to init Anthropic client: {e}", file=__import__("sys").stderr)

WORKDIR = Path.cwd()

TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"
TASKS_DIR = WORKDIR / ".tasks"
SKILLS_DIR = WORKDIR / "skills"
SESSIONS_DIR = WORKDIR / ".sessions"       # 🆕 树形 JSONL 会话存储目录
TRANSCRIPT_DIR = WORKDIR / ".transcripts"

TOKEN_THRESHOLD = 100000
POLL_INTERVAL = 5
IDLE_TIMEOUT = 60

VALID_MSG_TYPES = {"message", "broadcast", "shutdown_request",
                   "shutdown_response", "plan_approval_response"}

