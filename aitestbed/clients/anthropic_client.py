"""
Anthropic (Claude) client adapter for the 6G AI Traffic Testbed.

Uses the official Anthropic Python SDK (Messages API). The testbed's agent
loop exchanges plain-text messages and OpenAI-format tool definitions; this
adapter converts both to the Messages API format and back.

Environment variables:
  ANTHROPIC_API_KEY     API key from the Claude Console (required)
  ANTHROPIC_EFFORT      low | medium | high | xhigh | max (default: high)
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

    @property
    def provider(self) -> str:
        return "anthropic"

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
    def _convert_messages(messages: list[ChatMessage]) -> tuple[str, list[dict]]:
        """Split out the system prompt and merge consecutive same-role turns.

        The testbed sends tool calls and tool results as plain assistant and
        user text, sometimes several in a row; they are merged so that roles
        alternate.
        """
        system_parts: list[str] = []
        api_messages: list[dict] = []
        for m in messages:
            if m.role == MessageRole.SYSTEM:
                system_parts.append(m.content)
                continue
            if not m.content:
                continue
            role = "assistant" if m.role == MessageRole.ASSISTANT else "user"
            if api_messages and api_messages[-1]["role"] == role:
                api_messages[-1]["content"] += "\n\n" + m.content
            else:
                api_messages.append({"role": role, "content": m.content})
        return "\n\n".join(system_parts), api_messages

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
