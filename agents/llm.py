#!/usr/bin/env python3
"""LLM abstraction layer - unified interface for Anthropic and GLM."""

# Import the config MODULE (not the client variable directly)
# so we always read the latest client value via _get_client()
import agents.config as _cfg
import json

from .config import PROVIDER, MODEL, _init_provider


def _get_client():
    """Get the current LLM client, initializing lazily if needed."""
    _init_provider()
    return _cfg.client


# Ensure provider is initialized on import
_init_provider()


class ParsedToolUse:
    """Parsed tool_use content block."""
    def __init__(self, tool_id, name, arguments):
        self.type = "tool_use"
        self.id = tool_id
        self.name = name
        self.input = arguments


class ParsedText:
    """Parsed text content block."""
    def __init__(self, text):
        self.type = "text"
        self.text = text


class ParsedResponse:
    """Unified parsed LLM response."""
    def __init__(self):
        self.content = []
        self.stop_reason = "end_turn"


def call_llm(messages: list, tools: list = None, system: str = None,
             max_tokens: int = 8000):
    """Unified LLM call supporting both Anthropic and GLM providers."""
    if PROVIDER == "glm":
        glm_messages = []
        if system:
            glm_messages.append({"role": "system", "content": system})
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "tool_result":
                            text_parts.append(f"[Tool Result: {part.get('tool_use_id', '')}] {part.get('content', '')}")
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "\n".join(text_parts)
            glm_messages.append({"role": role, "content": str(content)})

        _client = _get_client()
        if not _client:
            raise RuntimeError(
                f"LLM client not initialized. PROVIDER={PROVIDER}, MODEL={MODEL}. "
                f"Check that GLM_API_KEY (for glm) or ANTHROPIC_BASE_URL + MODEL_ID (for anthropic) "
                f"are set correctly and the required package is installed (zai or anthropic)."
            )
        kwargs = {
            "model": MODEL,
            "messages": glm_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            glm_tools = []
            for t in tools:
                glm_tools.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {})
                    }
                })
            kwargs["tools"] = glm_tools
        return _client.chat.completions.create(**kwargs)
    else:
        _client = _get_client()
        if not _client:
            raise RuntimeError(
                f"LLM client not initialized. PROVIDER={PROVIDER}, MODEL={MODEL}. "
                f"Check that ANTHROPIC_BASE_URL and MODEL_ID are set, "
                f"and the 'anthropic' package is installed."
            )
        kwargs = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if system:
            kwargs["system"] = system
        return _client.messages.create(**kwargs)


def parse_llm_response(response):
    """Parse LLM response into unified ParsedResponse format."""
    if PROVIDER == "glm":
        choice = response.choices[0]
        message = choice.message
        parsed = ParsedResponse()
        parsed.stop_reason = "end_turn" if choice.finish_reason == "stop" else "tool_use"
        if hasattr(message, "content") and message.content:
            parsed.content.append(ParsedText(message.content))
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                parsed.content.append(ParsedToolUse(tc.id, tc.function.name, args))
            parsed.stop_reason = "tool_use"
        return parsed
    else:
        return response

