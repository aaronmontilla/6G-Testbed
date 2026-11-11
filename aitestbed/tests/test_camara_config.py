"""
Consistency of the CAMARA Dedicated Networks configuration: MCP server entry,
server group, rate limits, scenarios and anonymization aliases.

Run from aitestbed/:  python -m pytest tests
"""

import json
import pathlib

import pytest
import yaml

CONFIGS = pathlib.Path(__file__).resolve().parent.parent / "configs"

CAMARA_TOOLS = {
    "camara_list_profiles",
    "camara_get_profile",
    "camara_retrieve_service_areas",
    "camara_get_area",
    "camara_list_networks",
    "camara_get_network",
    "camara_create_network",
    "camara_delete_network",
    "camara_list_accesses",
    "camara_get_access",
    "camara_create_access",
    "camara_delete_access",
}
DN_SCENARIOS = ("dn_agent_baseline", "dn_agent_rule_in_system_prompt", "dn_cleanup")


@pytest.fixture(scope="module")
def mcp_config():
    return yaml.safe_load((CONFIGS / "mcp_servers.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scenarios():
    return yaml.safe_load((CONFIGS / "scenarios.yaml").read_text(encoding="utf-8"))[
        "scenarios"
    ]


def test_the_gateway_entry_lists_the_twelve_tools_and_runs_from_the_image(mcp_config):
    server = mcp_config["mcpServers"]["camara-dn"]
    assert set(server["tools"]) == CAMARA_TOOLS
    assert server["supports_http"] is True
    assert server["args"][-2:] == [
        "/opt/camara-venv/bin/python",
        "/opt/camara-dn/server.py",
    ]
    # Credentials are inherited from the environment, never written here.
    assert server["env"] == {}


def test_the_gateway_has_its_own_group_and_is_not_in_general(mcp_config):
    groups = mcp_config["serverGroups"]
    assert groups["dedicated_networks"] == ["camara-dn"]
    assert "camara-dn" not in groups["general"]


def test_every_camara_tool_shares_one_rate_budget(mcp_config):
    assert "camara_global" in mcp_config["rateGroups"]
    for tool in CAMARA_TOOLS:
        assert mcp_config["rateLimits"][tool] == {"rate_group": "camara_global"}


@pytest.mark.parametrize("name", DN_SCENARIOS)
def test_dedicated_network_scenarios_ship_disabled_and_use_the_gateway(scenarios, name):
    scenario = scenarios[name]
    assert scenario["disabled"] is True
    assert scenario["provider"] == "anthropic"
    assert scenario["server_group"] == "dedicated_networks"
    # The gateway runs as a local process: the agent-gateway hop is not shaped.
    assert scenario["mcp_transport"] == "stdio"


@pytest.mark.parametrize(
    "name", ["dn_agent_baseline", "dn_agent_rule_in_system_prompt"]
)
def test_agent_scenarios_carry_placeholders_not_environment_data(scenarios, name):
    prompts = scenarios[name]["prompts"]
    assert len(prompts) == 4
    for index in (0, 1, 3):  # P1, P2 and P4 depend on the CAMARA environment
        assert "<LAT>" in prompts[index] and "<LON>" in prompts[index]
    assert "<PHONE_E164>" in prompts[1]


def test_the_two_agent_scenarios_differ_only_in_the_system_prompt(scenarios):
    baseline = dict(scenarios["dn_agent_baseline"])
    with_rule = dict(scenarios["dn_agent_rule_in_system_prompt"])
    assert with_rule.pop("system_prompt").startswith(
        baseline.pop("system_prompt").rstrip()
    )
    baseline.pop("description")
    with_rule.pop("description")
    assert baseline == with_rule


def test_dedicated_network_scenarios_have_anonymization_aliases():
    aliases = json.loads(
        (CONFIGS / "anonymization_map.json").read_text(encoding="utf-8")
    )["scenarios"]
    for name in DN_SCENARIOS:
        assert name in aliases
