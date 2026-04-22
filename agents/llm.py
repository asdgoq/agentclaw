#!/usr/bin/env python3
"""LLM abstraction layer - unified interface for Anthropic and GLM."""

import json

# Import the config MODULE (not the client variable directly)
# so we always read the latest client value via _get_client()
import agents.config as _cfg
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


# ---------------------------------------------------------------------------
# Streaming support
# ---------------------------------------------------------------------------

def _build_glm_messages(messages: list, system: str = None) -> list:
    """Convert unified messages → GLM OpenAI-compatible format."""
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
                        text_parts.append(
                            f"[Tool Result: {part.get('tool_use_id', '')}] "
                            f"{part.get('content', '')}"
                        )
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "\n".join(text_parts)
        glm_messages.append({"role": role, "content": str(content)})
    return glm_messages


def _build_glm_tools(tools: list) -> list:
    """Convert unified tools → GLM function-calling format."""
    if not tools:
        return []
    glm_tools = []
    for t in tools:
        glm_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        })
    return glm_tools


def stream_call_llm(messages: list, tools: list = None, system: str = None,
                    max_tokens: int = 8000):
    """
    Streaming LLM call — yields (delta_text, event_type, event_data) tuples.

    Yields:
      - (text_chunk, "text_delta", None)       — each piece of streamed text
      - (None,          "tool_start", tool_info) — when a tool_use begins
      - (None,          "tool_delta", (id,name,args_chunk)) — tool arg streaming
      - (None,          "done", ParsedResponse)  — final response (always last)

    Usage:
        for chunk, evt, data in stream_call_llm(...):
            if evt == "text_delta":
                print(chunk, end="", flush=True)
            elif evt == "done":
                response = data
    """
    _client = _get_client()
    if not _client:
        raise RuntimeError(
            f"LLM client not initialized. PROVIDER={PROVIDER}, MODEL={MODEL}."
        )

    if PROVIDER == "glm":
        yield from _stream_glm(_client, messages, tools, system, max_tokens)
    else:
        yield from _stream_anthropic(_client, messages, tools, system, max_tokens)


def _stream_glm(client, messages: list, tools: list, system: str,
                max_tokens: int):
    """GLM (OpenAI-compatible) streaming."""
    glm_messages = _build_glm_messages(messages, system)
    kwargs = {
        "model": MODEL,
        "messages": glm_messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if tools:
        kwargs["tools"] = _build_glm_tools(tools)

    stream = client.chat.completions.create(**kwargs)

    # Accumulated state for building the final ParsedResponse
    content_buf = ""
    tool_calls_map: dict[int, dict] = {}   # index → {id, name, arguments}
    finish_reason = "stop"

    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        choice = chunk.choices[0]

        if choice.finish_reason:
            finish_reason = choice.finish_reason

        # Text delta
        if delta.content:
            content_buf += delta.content
            yield (delta.content, "text_delta", None)

        # Tool call deltas
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_map:
                    # New tool call starts
                    tc_id = tc.id or f"call_{idx}"
                    tc_name = tc.function.name or ""
                    tool_calls_map[idx] = {
                        "id": tc_id, "name": tc_name, "arguments": "",
                    }
                    yield (None, "tool_start", {
                        "id": tc_id, "name": tc_name,
                    })

                # Accumulate argument JSON string
                if tc.function.arguments:
                    tool_calls_map[idx]["arguments"] += tc.function.arguments
                    yield (None, "tool_delta",
                           (tc_id, tool_calls_map[idx]["name"],
                            tc.function.arguments))

    # Build final ParsedResponse
    parsed = ParsedResponse()
    parsed.stop_reason = "end_turn" if finish_reason == "stop" else "tool_use"

    if content_buf:
        parsed.content.append(ParsedText(content_buf))

    for tc_info in tool_calls_map.values():
        args_str = tc_info["arguments"]
        if isinstance(args_str, str):
            try:
                args = json.loads(args_str)
            except Exception:
                args = args_str or {}
        else:
            args = {}
        parsed.content.append(
            ParsedToolUse(tc_info["id"], tc_info["name"], args)
        )
        if tc_info["name"]:
            parsed.stop_reason = "tool_use"

    yield (None, "done", parsed)


def _stream_anthropic(client, messages: list, tools: list, system: str,
                      max_tokens: int):
    """Anthropic native streaming."""
    kwargs = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if tools:
        kwargs["tools"] = tools
    if system:
        kwargs["system"] = system

    with client.messages.stream(**kwargs) as stream:
        content_buf = ""
        current_tool = None   # accumulates current tool_use block
        stop_reason = "end_turn"

        for event in stream:
            event_type = event.type

            if event_type == "content_block_start":
                block = event.content_block
                if block and block.type == "tool_use":
                    current_tool = {
                        "id": block.id, "name": block.name,
                        "input_str": "", "input": {},
                    }
                    yield (None, "tool_start", {
                        "id": block.id, "name": block.name,
                    })

            elif event_type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    text = delta.text or ""
                    content_buf += text
                    yield (text, "text_delta", None)
                elif delta.type == "input_json_delta":
                    if current_tool is not None:
                        partial = delta.partial_json or ""
                        current_tool["input_str"] += partial
                        yield (None, "tool_delta",
                               (current_tool["id"], current_tool["name"],
                                partial))

            elif event_type == "content_block_stop":
                if current_tool is not None:
                    try:
                        current_tool["input"] = json.loads(
                            current_tool["input_str"]
                        )
                    except Exception:
                        current_tool["input"] = {}
                    current_tool = None

            elif event_type == "message_stop":
                pass

            elif event_type == "message_start":
                pass

            elif event_type == "message_delta":
                if hasattr(event, 'delta') and event.delta:
                    if hasattr(event.delta, 'stop_reason'):
                        sr = event.delta.stop_reason
                        if sr:
                            stop_reason = sr

        # Build final ParsedResponse
        parsed = ParsedResponse()
        parsed.stop_reason = stop_reason or "end_turn"

        if content_buf:
            parsed.content.append(ParsedText(content_buf))

        # Re-scan from the final message to get complete tool blocks
        # (the stream's final_message has all accumulated content)
        final = stream.get_final_message()
        if final and final.content:
            for block in final.content:
                if block.type == "tool_use":
                    parsed.content.append(ParsedToolUse(
                        block.id, block.name, block.input,
                    ))
                    parsed.stop_reason = "tool_use"

        yield (None, "done", parsed)

