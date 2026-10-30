# 6G AI Traffic Characterization Testbed

A framework for measuring and analyzing AI/LLM service traffic patterns under various network conditions, designed to support 3GPP SA4 6G Media Study contributions.

## Components

| Component | Description |
|-----------|-------------|
| [aitestbed/](./aitestbed/) | Main testing framework for running AI traffic experiments |
| [netemu/](./netemu/) | Network emulation library wrapping Linux tc/netem |

## aitestbed

The core testing framework that orchestrates experiments across multiple AI providers and scenarios.

**Features:**
- **11 scenario types**: Chat, agentic AI with MCP tools, image generation, multimodal, video understanding, realtime WebSocket/WebRTC
- **LLM providers**: OpenAI, Gemini, DeepSeek, Anthropic (Claude), vLLM, plus OpenAI Realtime (WebSocket/WebRTC)
- **60+ metrics**: TTFT/TTLT, latency percentiles, UL/DL ratios, token rates, agent loop factors
- **Multi-layer traffic capture**: L3/L4 via tcpdump, L7 via mitmproxy
- **SQLite logging** with structured metrics schema

```bash
# From the repo root:
pip install -e netemu
pip install -r aitestbed/requirements.txt
cd aitestbed
python orchestrator.py --scenario chat_basic --profile 5g_urban --runs 10
```

For the full SA4 cross-check run (all scenarios × profiles, with PCAP capture and report generation), see the **Cross-Checking for SA4 AI Traffic Characterization** section in `aitestbed/README.md`.

## netemu

A lightweight network emulation library providing a clean interface to Linux traffic control.

**Features:**
- Wraps `tc` and `netem` for delay, jitter, packet loss, and rate limiting
- Bidirectional shaping via IFB devices
- Predefined profiles including 3GPP 5QI mappings and SA4 S4-260848 reference conditions
- Context manager support for automatic cleanup

```python
from netemu import NetworkEmulator

with NetworkEmulator(interface="eth0") as emu:
    emu.apply_profile("5g_urban")  # 20ms delay, 0.1% loss, 100 Mbps
    # Run your tests here
# Rules automatically cleared
```

## Quick Start

```bash
# Clone and setup
cd testbed
python -m venv venv
source venv/bin/activate

# Install netemu first (separate package), then testbed dependencies
pip install -e netemu
pip install -r aitestbed/requirements.txt

# Optional, for L7 capture: mitmproxy in its own environment
# (it conflicts with requirements.txt; see aitestbed/requirements-l7.txt)
pipx install mitmproxy

# Set API keys
export OPENAI_API_KEY="your-key"

# Run a basic experiment
cd aitestbed
python orchestrator.py --scenario chat_basic --profile 6g_itu_hrllc --runs 5
```

## Docker

```bash
docker build -t 6g-ai-testbed -f aitestbed/Dockerfile .
docker run --cap-add=NET_ADMIN -e OPENAI_API_KEY="..." \
  6g-ai-testbed python orchestrator.py --scenario all --runs 10
```

## Network Profiles

Current test matrix (from `aitestbed/configs/profiles.yaml`, aligned with 3GPP SA4 S4-260848 Table C.Z-1):

| Profile | Delay | Jitter | Loss | Loss Distribution | Rate | Use Case |
|---------|-------|--------|------|-------------------|------|----------|
| `no_emulation` | 0 ms | 0 ms | 0% | -- | -- | Reference (no tc/netem) |
| `6g_itu_hrllc` | 1 ms | 0.2 ms | 0.001% | correlated (10%) | 300 Mbps | 6G HRLLC (ITU IMT-2030 / M.2160) |
| `5g_urban` | 20 ms | 5 ms | 0.1% | correlated (25%) | 100 Mbps | Mainstream urban cellular |
| `wifi_good` | 30 ms | 10 ms | 0.1% | correlated (30%) | 50 Mbps | Home/office WiFi |
| `cell_edge` | 120 ms | 30 ms | 1% | Gilbert-Elliot | 5 Mbps | Weak radio, heavy-tail jitter |
| `satellite_leo` | 22 ms | 7 ms DL / 8 ms UL | 0.5% DL / 0.8% UL | correlated (40% DL / 45% UL) | 100 DL / 15 UL Mbps | LEO satellite (asymmetric) |
| `satellite_geo` | 340 ms | 15 ms DL / 18 ms UL | 0.1% DL / 0.2% UL | correlated (20% DL / 25% UL) | 50 DL / 3 UL Mbps | GEO satellite (asymmetric) |
| `congested` | 200 ms | 50 ms | 3% | Gilbert-Elliot | 1 Mbps | Bufferbloat / heavy congestion |
| `5qi_7` | 80 ms | 10 ms | 0.1% | correlated (20%) | -- | 5QI 7: voice / live streaming |
| `5qi_80` | 8 ms | 1 ms | 1e-6 | correlated (5%) | -- | 5QI 80: low-latency eMBB / AR |

Asymmetric profiles (`satellite_leo`, `satellite_geo`) use an optional `uplink:` block that overrides egress-side fields. See [aitestbed/README.md](./aitestbed/README.md) for the full table with jitter, loss models, and advanced netem parameters.

## Requirements

- Python 3.10+
- Linux with `iproute2` (for network emulation)
- Sudo access or Docker with `NET_ADMIN` capability

