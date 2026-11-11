"""
Unit tests for native tool messages: the Anthropic adapter sends tool calls
and results as tool_use / tool_result blocks, and the agent loop builds the
messages that carry them.

No API key, MCP server or network access is needed. Run from aitestbed/:
python -m pytest tests
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

import clients.anthropic_client as ac
from clients.base import ChatMessage, ChatResponse, MessageRole, ToolCall
from scenarios.agent import GeneralAgentScenario

convert = ac.AnthropicClient._convert_messages


def block(**fields):
    """A fake SDK content block: attribute access plus to_dict()."""
    return SimpleNamespace(**fields, to_dict=lambda: dict(fields))


def test_tool_calls_and_results_become_blocks():
    messages = [
        ChatMessage(MessageRole.SYSTEM, "system prompt"),
        ChatMessage(MessageRole.USER, "list networks and profiles"),
        ChatMessage(
            MessageRole.ASSISTANT,
            "Checking both.",
            tool_calls=[
                ToolCall("toolu_1", "camara_list_networks", {}),
                ToolCall("toolu_2", "camara_list_profiles", {}),
            ],
        ),
        ChatMessage(MessageRole.TOOL, "[]", name="camara_list_networks", tool_call_id="toolu_1"),
        ChatMessage(
            MessageRole.TOOL,
            '{"error": "404", "success": false}',
            name="camara_list_profiles",
            tool_call_id="toolu_2",
            is_error=True,
        ),
    ]
    system, api_messages = convert(messages)
    assert system == "system prompt"
    assert api_messages == [
        {"role": "user", "content": "list networks and profiles"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking both."},
                {"type": "tool_use", "id": "toolu_1", "name": "camara_list_networks", "input": {}},
                {"type": "tool_use", "id": "toolu_2", "name": "camara_list_profiles", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "[]"},
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_2",
                    "content": '{"error": "404", "success": false}',
                    "is_error": True,
                },
            ],
        },
    ]


def test_raw_content_is_sent_back_unchanged():
    raw = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "tool_use", "id": "toolu_1", "name": "camara_get_area", "input": {"areaId": "a1"}},
    ]
    messages = [
        ChatMessage(MessageRole.USER, "area a1?"),
        ChatMessage(
            MessageRole.ASSISTANT,
            "",
            tool_calls=[ToolCall("toolu_1", "camara_get_area", {"areaId": "a1"})],
            raw_content=raw,
        ),
        ChatMessage(MessageRole.TOOL, "{}", name="camara_get_area", tool_call_id="toolu_1"),
    ]
    _, api_messages = convert(messages)
    assert api_messages[1] == {"role": "assistant", "content": raw}
    assert api_messages[2]["content"][0]["tool_use_id"] == "toolu_1"


def test_trimmed_history_keeps_calls_and_results_paired():
    messages = [
        ChatMessage(MessageRole.USER, "do it"),
        # The call that produced this result was trimmed away.
        ChatMessage(MessageRole.USER, "[3 earlier tool exchanges omitted for brevity]"),
        ChatMessage(MessageRole.TOOL, "old result", name="a", tool_call_id="toolu_old"),
        # This call lost its result.
        ChatMessage(MessageRole.ASSISTANT, "", tool_calls=[ToolCall("toolu_new", "b", {})]),
        ChatMessage(MessageRole.USER, "continue"),
    ]
    _, api_messages = convert(messages)
    assert api_messages[0] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "do it\n\n[3 earlier tool exchanges omitted for brevity]"},
            {"type": "text", "text": "Earlier tool result: old result"},
        ],
    }
    first, rest = api_messages[2]["content"][0], api_messages[2]["content"][1:]
    assert first["type"] == "tool_result" and first["tool_use_id"] == "toolu_new"
    assert first["is_error"] is True
    assert rest == [{"type": "text", "text": "continue"}]


def test_plain_text_history_is_unchanged():
    messages = [
        ChatMessage(MessageRole.USER, "do it"),
        ChatMessage(MessageRole.ASSISTANT, "Calling a..."),
        ChatMessage(MessageRole.USER, "Tool result from a: 1"),
    ]
    _, api_messages = convert(messages)
    assert api_messages == [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "Calling a..."},
        {"role": "user", "content": "Tool result from a: 1"},
    ]


class SequenceMessages:
    """Fake Messages API that returns canned responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **params):
        self.calls.append(params)
        return self.responses.pop(0)


def sdk_response(content, stop_reason):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        model="claude-sonnet-5-5",
        to_dict=lambda: {"stop_reason": stop_reason},
    )


@pytest.fixture
def claude(monkeypatch):
    def _make(responses, tool_messages="native"):
        monkeypatch.setenv("ANTHROPIC_FALLBACKS", "off")
        monkeypatch.setenv("ANTHROPIC_TOOL_MESSAGES", tool_messages)
        fake = SimpleNamespace(messages=SequenceMessages(responses))
        monkeypatch.setattr(ac.anthropic, "Anthropic", lambda **kwargs: fake)
        return ac.AnthropicClient(), fake.messages

    return _make


def test_tool_messages_setting(claude):
    assert claude([])[0].supports_native_tool_messages is True
    assert claude([], tool_messages="text")[0].supports_native_tool_messages is False


def test_assistant_tool_message_keeps_raw_blocks_without_nulls(claude):
    client, _ = claude([])
    response = ChatResponse(
        content="",
        tool_calls=[ToolCall("toolu_1", "camara_get_area", {})],
        raw_response=SimpleNamespace(
            content=[
                block(type="thinking", thinking="", signature="sig"),
                block(type="tool_use", id="toolu_1", name="camara_get_area", input={}, caller=None),
            ]
        ),
    )
    message = client.assistant_tool_message(response)
    assert message.role == MessageRole.ASSISTANT
    assert message.tool_calls[0].id == "toolu_1"
    assert message.raw_content == [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "tool_use", "id": "toolu_1", "name": "camara_get_area", "input": {}},
    ]


class FakeExecutor:
    def __init__(self, backend_meta=None):
        self.calls = []
        self.backend_meta = backend_meta

    def get_tools_for_openai(self):
        return [{"type": "function", "function": {"name": "camara_list_networks", "parameters": {}}}]

    async def execute(self, name, arguments):
        self.calls.append(name)
        return SimpleNamespace(
            success=False,
            result=None,
            error="CAMARA API error 404: not found",
            latency_sec=0.01,
            request_bytes=10,
            response_bytes=20,
            backend_meta=self.backend_meta,
        )


def run_turn(client, executor=None):
    logger = SimpleNamespace(records=[])
    logger.log = logger.records.append
    scenario = GeneralAgentScenario(client, logger, {"model": "claude-sonnet-5-5"})
    scenario.tool_executor = executor or FakeExecutor()
    scenario.tool_executor.records = logger.records
    result = asyncio.run(
        scenario._run_agent_turn(
            user_prompt="list my networks",
            system_prompt="system prompt",
            model="claude-sonnet-5-5",
            session_id="s1",
            turn_index=0,
            run_index=0,
            network_profile="no_emulation",
        )
    )
    return result, scenario.tool_executor


def test_agent_loop_sends_native_tool_messages(claude):
    client, api = claude(
        [
            sdk_response(
                [block(type="tool_use", id="toolu_1", name="camara_list_networks", input={})],
                "tool_use",
            ),
            sdk_response([block(type="text", text="You have no networks.")], "end_turn"),
        ]
    )
    result, executor = run_turn(client)
    assert executor.calls == ["camara_list_networks"]
    assert result["api_calls"] == 2
    second = api.calls[1]["messages"]
    assert second[1] == {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "camara_list_networks", "input": {}}],
    }
    assert second[2]["role"] == "user"
    tool_result = second[2]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "toolu_1"
    assert tool_result["is_error"] is True
    assert "404" in tool_result["content"]


def test_agent_loop_text_mode_keeps_plain_text(claude):
    client, api = claude(
        [
            sdk_response(
                [block(type="tool_use", id="toolu_1", name="camara_list_networks", input={})],
                "tool_use",
            ),
            sdk_response([block(type="text", text="Done.")], "end_turn"),
        ],
        tool_messages="text",
    )
    run_turn(client)
    second = api.calls[1]["messages"]
    assert second[1] == {"role": "assistant", "content": "Calling camara_list_networks..."}
    assert second[2]["content"].startswith("Tool result from camara_list_networks:")


def test_gateway_telemetry_is_stored_with_the_tool_call(claude):
    """The CAMARA gateway reports its own time and connections in _meta (backend_meta)."""
    meta = {
        "backend_calls": [{"method": "GET", "latency_sec": 0.1, "new_connection": False}],
        "backend_total_latency_sec": 0.1,
        "gateway_latency_sec": 0.1004,
        "token_wait_sec": 0.0,
        "backend_new_connections": 0,
        "backend_connect_sec": 0.0,
    }
    client, _ = claude(
        [
            sdk_response(
                [block(type="tool_use", id="toolu_1", name="camara_list_networks", input={})],
                "tool_use",
            ),
            sdk_response([block(type="text", text="Done.")], "end_turn"),
        ]
    )
    _, executor = run_turn(client, FakeExecutor(backend_meta=meta))
    tool_records = [json.loads(r.metadata) for r in executor.records
                    if r.metadata and json.loads(r.metadata).get("type") == "mcp_tool_call"]
    assert len(tool_records) == 1
    stored = tool_records[0]
    for key in ("backend_total_latency_sec", "gateway_latency_sec", "token_wait_sec",
                "backend_new_connections", "backend_connect_sec", "backend_calls"):
        assert stored[key] == meta[key]
