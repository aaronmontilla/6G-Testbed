# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Anthropic (Claude) LLM provider (`provider: "anthropic"`), using the
  official Anthropic SDK: chat, streaming and tool calling for agent
  scenarios; `chat_anthropic` and `chat_anthropic_streaming` scenarios;
  anonymization aliases for Claude models.
- CAMARA Dedicated Networks agent scenarios (`dn_agent_baseline`,
  `dn_agent_rule_in_system_prompt`, `dn_cleanup`; disabled until configured)
  over the CAMARA Dedicated Networks MCP gateway: `camara-dn` MCP server
  entry, `dedicated_networks` server group, shared rate budget, optional
  installation of the gateway in the image (`--build-arg WITH_CAMARA=1`,
  pinned commit), `CAMARA_*` variables in `.env.example` and
  `docker-compose.yml`, configuration consistency tests.

### Changed
- The CAMARA gateway is pinned to `c5d2505`: it reuses its connection to the
  CAMARA API between tool calls and reports per-call telemetry in `_meta`
  (backend calls, token time, time inside the server, connections opened).
  The agent scenarios now also store `gateway_latency_sec`, `token_wait_sec`,
  `backend_new_connections` and `backend_connect_sec` with each tool call.
- The `dn_*` CAMARA scenarios use `mcp_transport: stdio`. With HTTP the network
  profile was applied twice to every tool call: on the loopback hop between the
  agent and the gateway, and again on the gateway's calls to the CAMARA API.
  The gateway is a local process of the agent host, so only the second hop
  (and the model API calls) should cross the emulated link.

### Fixed
- Payload traces (`TRACE_PAYLOADS=1`) written in the same second by the same
  session and turn shared one file name, so later traces overwrote earlier
  ones (an agent turn writes a trace per model call and per tool call). File
  names now include milliseconds, are created exclusively, and take a `_1`,
  `_2`, ... suffix when the name is taken; no trace is overwritten.
- The Docker entrypoint asked for `OPENAI_API_KEY` for every scenario whose
  name contains "chat", including `chat_anthropic`; Claude scenarios now
  require `ANTHROPIC_API_KEY` instead.

## [0.1.0] - 2026-01-19
- Initial release of the 6G AI Traffic Characterization Testbed.
