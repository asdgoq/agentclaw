#!/usr/bin/env python3
"""LLM 抽象层 - Anthropic 和 GLM 的统一接口。

包含自动 429 限流重试和指数退避。
"""

import json
import threading
import time

# 导入 config 模块（而非直接导入 client 变量）
# 以便始终通过 _get_client() 读取最新的 client 值
import agents.core.config as _cfg
from ..core.config import PROVIDER, MODEL, _init_provider

# ------------------------------------------------------------------
# 全局限流器 + 重试配置
# ------------------------------------------------------------------
# 信号量，限制并发 LLM API 调用数（防止 429）
_LLM_CALL_SEMAPHORE = threading.Semaphore(5)  # 最多 5 个并发 API 调用

# 429 错误的重试配置
_MAX_RETRIES = 5          # 429 时的最大重试次数
_INITIAL_BACKOFF = 2.0    # 初始退避时间（秒）
_MAX_BACKOFF = 60.0       # 最大退避时间上限


def _get_client():
    """获取当前 LLM 客户端，如未初始化则延迟初始化。"""
    _init_provider()
    return _cfg.client


# 确保导入时初始化 provider
_init_provider()


class ParsedToolUse:
    """解析后的 tool_use 内容块。"""
    def __init__(self, tool_id, name, arguments):
        self.type = "tool_use"
        self.id = tool_id
        self.name = name
        self.input = arguments


class ParsedText:
    """解析后的文本内容块。"""
    def __init__(self, text):
        self.type = "text"
        self.text = text


class ParsedResponse:
    """统一的解析后 LLM 响应。"""
    def __init__(self):
        self.content = []
        self.stop_reason = "end_turn"


def call_llm(messages: list, tools: list = None, system: str = None,
             max_tokens: int = 8000):
    """统一的 LLM 调用，支持 Anthropic 和 GLM 两种提供者。

    遇到 429（限流）时自动重试，使用指数退避。
    使用信号量限制并发调用数。
    """
    last_exception = None
    backoff = _INITIAL_BACKOFF

    for attempt in range(_MAX_RETRIES + 1):
        # 限流：等待获取调用槽位后再调用 API
        with _LLM_CALL_SEMAPHORE:
            try:
                return _do_call_llm(messages, tools, system, max_tokens)
            except Exception as e:
                last_exception = e
                err_str = str(e).lower()
                # 检查是否为 429 限流错误
                if "429" in err_str or "rate" in err_str or "reachlimit" in err_str or "1302" in err_str:
                    if attempt < _MAX_RETRIES:
                        print(f"  [LLM] 429 rate limit hit (attempt {attempt+1}/{_MAX_RETRIES+1}), "
                              f"retrying in {backoff:.1f}s...")
                        time.sleep(backoff)
                        backoff = min(backoff * 2, _MAX_BACKOFF)
                        continue
                # 非 429 错误或重试次数用尽——抛出异常
                raise

    # 理论上不应到达此处，但以防万一
    raise last_exception


def _do_call_llm(messages: list, tools: list = None, system: str = None,
                 max_tokens: int = 8000):
    """实际的 LLM 调用实现（无重试逻辑）。"""
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
    """将 LLM 响应解析为统一的 ParsedResponse 格式。"""
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
# 流式支持
# ---------------------------------------------------------------------------

def _build_glm_messages(messages: list, system: str = None) -> list:
    """将统一消息格式转换为 GLM OpenAI 兼容格式。"""
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
    """将统一工具格式转换为 GLM function-calling 格式。"""
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
    流式 LLM 调用——生成 (delta_text, event_type, event_data) 元组。

    生成内容:
      - (text_chunk, "text_delta", None)       — 每段流式文本
      - (None,          "tool_start", tool_info) — 当 tool_use 开始时
      - (None,          "tool_delta", (id,name,args_chunk)) — 工具参数流式传输
      - (None,          "done", ParsedResponse)  — 最终响应（总是最后一条）

    用法:
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
    """GLM（OpenAI 兼容）流式调用。"""
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

    # 用于构建最终 ParsedResponse 的累积状态
    content_buf = ""
    tool_calls_map: dict[int, dict] = {}   # 索引 → {id, name, arguments}
    finish_reason = "stop"

    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        choice = chunk.choices[0]

        if choice.finish_reason:
            finish_reason = choice.finish_reason

        # 文本增量
        if delta.content:
            content_buf += delta.content
            yield (delta.content, "text_delta", None)

        # 工具调用增量
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_map:
                    # 新的工具调用开始
                    tc_id = tc.id or f"call_{idx}"
                    tc_name = tc.function.name or ""
                    tool_calls_map[idx] = {
                        "id": tc_id, "name": tc_name, "arguments": "",
                    }
                    yield (None, "tool_start", {
                        "id": tc_id, "name": tc_name,
                    })

                # 累积参数 JSON 字符串
                if tc.function.arguments:
                    tool_calls_map[idx]["arguments"] += tc.function.arguments
                    yield (None, "tool_delta",
                           (tc_id, tool_calls_map[idx]["name"],
                            tc.function.arguments))

    # 构建最终 ParsedResponse
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
    """Anthropic 原生流式调用。"""
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
        current_tool = None   # 累积当前 tool_use 块
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

        # 构建最终 ParsedResponse
        parsed = ParsedResponse()
        parsed.stop_reason = stop_reason or "end_turn"

        if content_buf:
            parsed.content.append(ParsedText(content_buf))

        # 从最终消息中重新扫描以获取完整的工具块
        # （流的 final_message 包含所有累积的内容）
        final = stream.get_final_message()
        if final and final.content:
            for block in final.content:
                if block.type == "tool_use":
                    parsed.content.append(ParsedToolUse(
                        block.id, block.name, block.input,
                    ))
                    parsed.stop_reason = "tool_use"

        yield (None, "done", parsed)

