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

### Fixed
- The Docker entrypoint asked for `OPENAI_API_KEY` for every scenario whose
  name contains "chat", including `chat_anthropic`; Claude scenarios now
  require `ANTHROPIC_API_KEY` instead.

## [0.1.0] - 2026-01-19
- Initial release of the 6G AI Traffic Characterization Testbed.
