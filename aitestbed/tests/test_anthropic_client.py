"""
Unit tests for the Anthropic (Claude) client adapter.

The Anthropic SDK client is replaced by a fake, so no API key or network
access is needed. Run from aitestbed/:  python -m pytest tests
"""

from types import SimpleNamespace

import pytest

import clients.anthropic_client as ac
from clients.base import ChatMessage, MessageRole


class FakeMessages:
    """Records the parameters of each create() call and returns a canned response."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **params):
        self.calls.append(params)
        return self.response


class FakeAnthropic:
    def __init__(self, response):
        self.messages = FakeMessages(response)
        self.beta = SimpleNamespace(messages=FakeMessages(response))


def make_response(content, stop_reason="end_turn", stop_details=None):
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        cache_creation_input_tokens=10,
        cache_read_input_tokens=5,
    )
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=usage,
        model="claude-sonnet-5-5",
        to_dict=lambda: {"stop_reason": stop_reason},
    )


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(block_id, name, arguments):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=arguments)


@pytest.fixture
def make_client(monkeypatch):
    def _make(response, fallbacks="default", effort="high"):
        monkeypatch.setenv("ANTHROPIC_FALLBACKS", fallbacks)
        monkeypatch.setenv("ANTHROPIC_EFFORT", effort)
        fake = FakeAnthropic(response)
        monkeypatch.setattr(ac.anthropic, "Anthropic", lambda **kwargs: fake)
        return ac.AnthropicClient(), fake

    return _make


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "camara_get_area",
            "description": "Get one service area",
            "parameters": {
                "type": "object",
                "properties": {"areaId": {"type": "string"}},
            },
        },
    }
]


def test_tools_are_converted_to_messages_api_format():
    converted = ac.AnthropicClient._convert_tools(TOOLS)
    assert converted == [
        {
            "name": "camara_get_area",
            "description": "Get one service area",
            "input_schema": {
                "type": "object",
                "properties": {"areaId": {"type": "string"}},
            },
        }
    ]


def test_messages_split_system_and_merge_consecutive_roles():
    messages = [
        ChatMessage(MessageRole.SYSTEM, "system prompt"),
        ChatMessage(MessageRole.USER, "do it"),
        ChatMessage(MessageRole.ASSISTANT, "Calling a..."),
        ChatMessage(MessageRole.ASSISTANT, "Calling b..."),
        ChatMessage(MessageRole.USER, "Tool result from a: 1"),
        ChatMessage(MessageRole.USER, ""),
        ChatMessage(MessageRole.USER, "Tool result from b: 2"),
    ]
    system, api_messages = ac.AnthropicClient._convert_messages(messages)
    assert system == "system prompt"
    assert api_messages == [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "Calling a...\n\nCalling b..."},
        {"role": "user", "content": "Tool result from a: 1\n\nTool result from b: 2"},
    ]


def test_build_params_sets_effort_cache_and_tools(make_client):
    client, _ = make_client(make_response([text_block("ok")]), effort="medium")
    params = client._build_params(
        [ChatMessage(MessageRole.USER, "hi")], "claude-sonnet-5-5", TOOLS
    )
    assert params["output_config"] == {"effort": "medium"}
    assert params["cache_control"] == {"type": "ephemeral"}
    assert params["tools"][0]["name"] == "camara_get_area"
    assert "system" not in params


def test_chat_returns_text_tool_calls_and_tokens(make_client):
    response = make_response(
        [
            text_block("Let me check."),
            tool_block("toolu_1", "camara_get_area", {"areaId": "a1"}),
        ],
        stop_reason="tool_use",
    )
    client, _ = make_client(response)
    result = client.chat(
        [ChatMessage(MessageRole.USER, "area a1?")], "claude-sonnet-5-5", tools=TOOLS
    )
    assert result.content == "Let me check."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "camara_get_area"
    assert result.tool_calls[0].arguments == {"areaId": "a1"}
    assert result.tokens_in == 115  # input + cache creation + cache read
    assert result.tokens_out == 20
    assert result.model == "claude-sonnet-5-5"


def test_refusal_is_reported_as_text(make_client):
    details = SimpleNamespace(category="cyber", explanation="declined")
    client, _ = make_client(
        make_response([], stop_reason="refusal", stop_details=details)
    )
    result = client.chat([ChatMessage(MessageRole.USER, "x")], "claude-sonnet-5-5")
    assert result.content == "[Claude refusal: cyber]"
    assert result.tool_calls == []


def test_fallbacks_use_beta_endpoint_on_supported_models(make_client):
    client, fake = make_client(make_response([text_block("ok")]), fallbacks="default")
    client.chat([ChatMessage(MessageRole.USER, "x")], "claude-sonnet-5-5")
    assert fake.messages.calls == []
    call = fake.beta.messages.calls[0]
    assert call["betas"] == [ac.FALLBACK_BETA]
    assert call["extra_body"] == {"fallbacks": "default"}


def test_fallbacks_off_or_unsupported_model_use_plain_endpoint(make_client):
    client, fake = make_client(make_response([text_block("ok")]), fallbacks="off")
    client.chat([ChatMessage(MessageRole.USER, "x")], "claude-sonnet-5-5")
    assert len(fake.messages.calls) == 1 and fake.beta.messages.calls == []

    client, fake = make_client(make_response([text_block("ok")]), fallbacks="default")
    client.chat([ChatMessage(MessageRole.USER, "x")], "claude-haiku-5-5")
    assert len(fake.messages.calls) == 1 and fake.beta.messages.calls == []
