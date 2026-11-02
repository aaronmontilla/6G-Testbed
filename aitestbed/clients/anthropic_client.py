"""
Anthropic (Claude) client adapter for the 6G AI Traffic Testbed.

Uses the official Anthropic Python SDK (Messages API). The adapter converts
the testbed's messages and OpenAI-format tool definitions to the Messages
API format and back. In the agent loop, tool calls and tool results travel
as native tool_use / tool_result blocks, so the model sees each result as
the answer to its own call (with plain text, Claude may not recognize the
results as its own and repeat the calls).

Environment variables:
  ANTHROPIC_API_KEY     API key from the Claude Console (required)
  ANTHROPIC_EFFORT      low | medium | high | xhigh | max (default: high)
  ANTHROPIC_TOOL_MESSAGES
                        native | text (default: native). "text" restores the
                        plain-text exchange used by the other providers.
  ANTHROPIC_FALLBACKS   "default" enables server-side refusal fallbacks on
                        the models that support them; "off" disables them
                        (default: default). With fallbacks on, a declined
                        request may be served by another model: the model
                        that answered is recorded in ChatResponse.model and
                        in the logged response payload.
"""

import os
import time
from typing import Iterator, Optional

import anthropic

from .base import (
    LLMClient,
    ChatMessage,
    ChatResponse,
    StreamingResponse,
    ToolCall,
    MessageRole,
    estimate_payload_bytes,
)

DEFAULT_MAX_TOKENS = 16000

# Models that accept the server-side `fallbacks: "default"` parameter.
FALLBACK_MODELS = {
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-fable-5-1",
    "claude-sonnet-5-5",
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicClient(LLMClient):
    """Claude client adapter with traffic metrics collection."""

    def __init__(self, api_key: Optional[str] = None):
        # Without an explicit key the SDK resolves ANTHROPIC_API_KEY (or an
        # `ant auth login` profile) from the environment.
        self.client = (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )
        self.effort = os.environ.get("ANTHROPIC_EFFORT", "high")
        self.fallbacks = os.environ.get("ANTHROPIC_FALLBACKS", "default").lower()
        self.supports_native_tool_messages = (
            os.environ.get("ANTHROPIC_TOOL_MESSAGES", "native").lower() != "text"
        )

    @property
    def provider(self) -> str:
        return "anthropic"

    def assistant_tool_message(self, response: ChatResponse) -> ChatMessage:
        """Keep the response's content blocks (thinking, text, tool_use) so
        they are sent back unchanged on the next request."""
        message = super().assistant_tool_message(response)
        raw_blocks = getattr(response.raw_response, "content", None)
        if raw_blocks:
            message.raw_content = [
                {k: v for k, v in block.to_dict().items() if v is not None}
                for block in raw_blocks
            ]
        return message

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _convert_tools(tools: Optional[list[dict]]) -> list[dict]:
        """OpenAI function format -> Messages API tool definitions."""
        converted = []
        for tool in tools or []:
            fn = tool.get("function", tool)
            converted.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description") or "",
                    "input_schema": fn.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
        return converted

    @staticmethod
    def _content_of(m: ChatMessage) -> str | list[dict] | None:
        """Message content: a string for plain text, blocks for tool traffic."""
        if m.role == MessageRole.TOOL:
            block = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": m.content,
            }
            if m.is_error:
                block["is_error"] = True
            return [block]
        if m.role == MessageRole.ASSISTANT and m.raw_content:
            return [dict(b) for b in m.raw_content]
        if m.role == MessageRole.ASSISTANT and m.tool_calls:
            blocks = [{"type": "text", "text": m.content}] if m.content else []
            blocks += [
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                for tc in m.tool_calls
            ]
            return blocks
        return m.content or None

    @staticmethod
    def _as_blocks(content: str | list[dict]) -> list[dict]:
        return [{"type": "text", "text": content}] if isinstance(content, str) else content

    @staticmethod
    def _pair_tool_blocks(api_messages: list[dict]) -> list[dict]:
        """Make every tool_use answered by a tool_result in the next message.

        Trimming long histories can separate a call from its result. A result
        whose call is gone becomes plain text; a call whose result is gone
        gets a placeholder error result. Results always lead their message.
        """
        def ids(message, block_type, key):
            if not message or not isinstance(message["content"], list):
                return set()
            return {b[key] for b in message["content"] if b.get("type") == block_type}

        out: list[dict] = []
        for msg in api_messages:
            prev = out[-1] if out and out[-1]["role"] == "assistant" else None
            if msg["role"] == "user":
                calls = ids(prev, "tool_use", "id")
                blocks = AnthropicClient._as_blocks(msg["content"])
                results, others = [], []
                for b in blocks:
                    if b.get("type") != "tool_result":
                        others.append(b)
                    elif b["tool_use_id"] in calls:
                        results.append(b)
                    else:
                        others.append({"type": "text", "text": f"Earlier tool result: {b['content']}"})
                answered = {b["tool_use_id"] for b in results}
                results += [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": "Result omitted to keep the conversation short.",
                        "is_error": True,
                    }
                    for call_id in sorted(calls - answered)
                ]
                if results or not isinstance(msg["content"], str):
                    msg = {"role": "user", "content": results + others}
            out.append(msg)
        return out

    @staticmethod
    def _convert_messages(messages: list[ChatMessage]) -> tuple[str, list[dict]]:
        """Split out the system prompt and merge consecutive same-role turns.

        Plain-text turns (and, with ANTHROPIC_TOOL_MESSAGES=text, the tool
        calls and results the testbed sends as text) are merged so that roles
        alternate; tool traffic is kept as tool_use / tool_result blocks.
        """
        system_parts: list[str] = []
        api_messages: list[dict] = []
        for m in messages:
            if m.role == MessageRole.SYSTEM:
                system_parts.append(m.content)
                continue
            content = AnthropicClient._content_of(m)
            if not content:
                continue
            role = "assistant" if m.role == MessageRole.ASSISTANT else "user"
            if api_messages and api_messages[-1]["role"] == role:
                prev = api_messages[-1]["content"]
                if isinstance(prev, str) and isinstance(content, str):
                    api_messages[-1]["content"] = prev + "\n\n" + content
                else:
                    api_messages[-1]["content"] = (
                        AnthropicClient._as_blocks(prev) + AnthropicClient._as_blocks(content)
                    )
            else:
                api_messages.append({"role": role, "content": content})
        return "\n\n".join(system_parts), AnthropicClient._pair_tool_blocks(api_messages)

    def _build_params(
        self,
        messages: list[ChatMessage],
        model: str,
        tools: Optional[list[dict]] = None,
    ) -> dict:
        system, api_messages = self._convert_messages(messages)
        params = {
            "model": model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "messages": api_messages,
            # Caches tools + system + history prefix across agent iterations.
            "cache_control": {"type": "ephemeral"},
        }
        if not model.startswith("claude-haiku-4"):
            params["output_config"] = {"effort": self.effort}
        if system:
            params["system"] = system
        if tools:
            params["tools"] = self._convert_tools(tools)
        return params

    def _use_fallbacks(self, model: str) -> bool:
        return self.fallbacks == "default" and model in FALLBACK_MODELS

    def _create(self, params: dict):
        if self._use_fallbacks(params["model"]):
            return self.client.beta.messages.create(
                betas=[FALLBACK_BETA], extra_body={"fallbacks": "default"}, **params
            )
        return self.client.messages.create(**params)

    def _stream(self, params: dict):
        if self._use_fallbacks(params["model"]):
            return self.client.beta.messages.stream(
                betas=[FALLBACK_BETA], extra_body={"fallbacks": "default"}, **params
            )
        return self.client.messages.stream(**params)

    @staticmethod
    def _usage_tokens(usage) -> tuple[Optional[int], Optional[int]]:
        if usage is None:
            return None, None
        tokens_in = (
            (usage.input_tokens or 0)
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            + (getattr(usage, "cache_read_input_tokens", 0) or 0)
        )
        return tokens_in, usage.output_tokens

    @staticmethod
    def _text_of(response) -> str:
        text = "".join(b.text for b in response.content if b.type == "text")
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            text = text or f"[Claude refusal: {category or 'unspecified'}]"
        return text

    # --------------------------------------------------------------- interface

    def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        stream: bool = False,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> ChatResponse | Iterator[str]:
        if stream:
            return self._chat_stream_generator(messages, model)
        return self._chat_sync(messages, model, tools)

    def _chat_sync(
        self,
        messages: list[ChatMessage],
        model: str,
        tools: Optional[list[dict]] = None,
    ) -> ChatResponse:
        params = self._build_params(messages, model, tools)
        request_bytes = estimate_payload_bytes(params)
        request_payload = {"format": "anthropic.messages.create", "payload": params}

        t_start = time.time()
        response = self._create(params)
        t_end = time.time()

        content = self._text_of(response)
        tool_calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in response.content
            if b.type == "tool_use"
        ]

        response_dump = response.to_dict()
        tokens_in, tokens_out = self._usage_tokens(response.usage)
        if tokens_in is None:
            tokens_in = self.estimate_message_tokens(messages, model)
        if tokens_out is None:
            tokens_out = self.estimate_tokens(content, model)

        return ChatResponse(
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            total_tokens=tokens_in + tokens_out,
            latency_sec=t_end - t_start,
            model=response.model,
            tool_calls=tool_calls,
            raw_response=response,
            request_bytes=request_bytes,
            response_bytes=estimate_payload_bytes(response_dump),
            request_payload=request_payload,
            response_payload={
                "format": "anthropic.messages.response",
                "payload": response_dump,
            },
        )

    def _chat_stream_generator(
        self, messages: list[ChatMessage], model: str
    ) -> Iterator[str]:
        params = self._build_params(messages, model)
        with self._stream(params) as stream:
            for text in stream.text_stream:
                yield text

    def chat_streaming(
        self, messages: list[ChatMessage], model: str, **kwargs
    ) -> StreamingResponse:
        params = self._build_params(messages, model)

        streaming_response = StreamingResponse()
        streaming_response.t_request_start = time.time()
        streaming_response.request_bytes = estimate_payload_bytes(params)
        streaming_response.model = model
        streaming_response.request_payload = {
            "format": "anthropic.messages.stream",
            "payload": params,
        }

        with self._stream(params) as stream:
            for text in stream.text_stream:
                streaming_response.add_chunk(text, timestamp=time.time())
            final = stream.get_final_message()

        tokens_in, tokens_out = self._usage_tokens(final.usage)
        streaming_response.tokens_in = tokens_in or self.estimate_message_tokens(
            messages, model
        )
        streaming_response.tokens_out = tokens_out or self.estimate_tokens(
            streaming_response.total_content, model
        )
        streaming_response.model = final.model
        streaming_response.response_payload = {
            "format": "anthropic.messages.response_summary",
            "payload": {
                "content": streaming_response.total_content,
                "stop_reason": final.stop_reason,
                "model": final.model,
                "tokens_in": streaming_response.tokens_in,
                "tokens_out": streaming_response.tokens_out,
            },
        }
        return streaming_response
