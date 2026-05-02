"""
LLM client using OpenAI-compatible API.

Supports: DeepSeek, OpenAI, Qwen, Ollama, vLLM, and any OpenAI-compatible endpoint.
Configure via:
  OPENAI_API_KEY   - API key
  OPENAI_BASE_URL  - API base URL
  OPENAI_MODEL     - Model name
"""
import json
import os
from typing import Any, Optional

from openai import OpenAI

from ..trajectory.schemas import LLMResponse
# 默认使用deepseek-chat模型和deepseek的API端点，用户也可以通过环境变量覆盖这些默认值以使用其他兼容OpenAI API的服务或模型。
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

# 转化工具描述
def _convert_tool_definitions(agent_tools: list[dict]) -> list[dict]:
    """Convert Anthropic-style tool format to OpenAI function-calling format."""
    openai_tools = []
    for t in agent_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema",
                                    {"type": "object", "properties": {}}),
            }
        })
    return openai_tools

# 转化消息格式
def _convert_messages(messages: list[dict]) -> list[dict]:
    """Convert messages with Anthropic-style content blocks to OpenAI format."""
    converted = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "assistant":
            if isinstance(content, list):
                text_parts = []
                tool_calls = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "thinking":
                        pass
                    elif btype == "tool_use":
                        tool_calls.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(
                                    block["input"], ensure_ascii=False
                                ),
                            }
                        })
                out: dict[str, Any] = {"role": "assistant"}
                if text_parts:
                    out["content"] = "\n".join(text_parts)
                if tool_calls:
                    out["tool_calls"] = tool_calls
                converted.append(out)
            else:
                converted.append({"role": "assistant", "content": content or ""})

        elif role == "user":
            if isinstance(content, list):
                tool_results = [
                    b for b in content
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                ]
                if tool_results:
                    for tr in tool_results:
                        inner = tr.get("content", [])
                        if isinstance(inner, list):
                            result_text = "\n".join(
                                c.get("text", "") for c in inner
                                if isinstance(c, dict)
                            )
                        else:
                            result_text = str(inner)
                        converted.append({
                            "role": "tool",
                            "tool_call_id": tr["tool_use_id"],
                            "content": result_text,
                        })
                else:
                    text = "\n".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                    converted.append({"role": "user", "content": text})
            else:
                converted.append({"role": "user", "content": content})
        else:
            converted.append(msg)
    return converted


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 16000,
    ):
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "No API key found. Set OPENAI_API_KEY environment variable.\n"
                "For DeepSeek: https://platform.deepseek.com/api_keys\n"
                "For custom endpoints, also set OPENAI_BASE_URL."
            )

        resolved_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", DEFAULT_BASE_URL
        )
        self.client = OpenAI(api_key=resolved_key, base_url=resolved_url)
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens

    def complete(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        tools: Optional[list[dict]] = None,
    ) -> LLMResponse:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        openai_messages = _convert_messages(all_messages)
        payload_size = len(json.dumps(openai_messages, ensure_ascii=False))
        print(
            f"  [LLM] model={self.model} payload={payload_size}chars "
            f"(~{payload_size // 4} tokens) msgs={len(openai_messages)}",
            flush=True,
        )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = _convert_tool_definitions(tools)
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message
        finish_reason = choice.finish_reason

        usage = {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }

        content_blocks = []
        if msg.content:
            content_blocks.append({"type": "text", "text": msg.content})

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": args,
                })

        if finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif finish_reason in ("stop", "end_turn"):
            stop_reason = "end_turn"
        elif finish_reason == "length":
            # Response was truncated by max_tokens — the LLM may have more to say
            # or more tools to call. Treat as tool_use to force another round,
            # letting the LLM explicitly decide to stop.
            print(
                f"  [LLM] WARNING: response truncated (max_tokens={self.max_tokens}). "
                f"Consider increasing MAX_TOKENS.",
                flush=True,
            )
            stop_reason = "tool_use"
        else:
            # "content_filter" or unknown — stop to avoid infinite loop
            stop_reason = "end_turn"

        return LLMResponse(
            content=content_blocks,
            stop_reason=stop_reason,
            usage=usage,
            model=self.model,
        )
