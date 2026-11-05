"""Trace files: one file per trace, none overwritten."""

import json
import time
from pathlib import Path

from analysis.trace_logger import TraceLogger


def _write(logger, turn_index=0, note="n"):
    return logger.write_trace(
        scenario_id="dn_agent_baseline",
        session_id="12345678-aaaa-bbbb-cccc-000000000000",
        turn_index=turn_index,
        run_index=0,
        network_profile="5g_urban",
        provider="anthropic",
        model="m",
        request_payload={"note": note},
        note=note,
    )


def test_traces_of_the_same_turn_and_second_get_separate_files(tmp_path, monkeypatch):
    frozen = time.time()
    monkeypatch.setattr(time, "time", lambda: frozen)
    logger = TraceLogger(log_dir=tmp_path, enabled=True)

    paths = [_write(logger, note=f"call {i}") for i in range(5)]

    assert len(set(paths)) == 5
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        Path(p).name for p in paths
    )
    notes = {json.loads(open(p, encoding="utf-8").read())["note"] for p in paths}
    assert notes == {f"call {i}" for i in range(5)}


def test_file_name_has_milliseconds_scenario_session_and_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000.123)
    logger = TraceLogger(log_dir=tmp_path, enabled=True)

    name = Path(_write(logger, turn_index=7)).name

    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(1_700_000_000.123))
    assert name == f"{stamp}_123_dn_agent_baseline_12345678_turn007.json"


def test_an_existing_file_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000.0)
    first = TraceLogger(log_dir=tmp_path, enabled=True)
    second = TraceLogger(log_dir=tmp_path, enabled=True)  # e.g. another scenario instance

    a = _write(first, note="first")
    b = _write(second, note="second")

    assert a != b
    assert json.loads(open(a, encoding="utf-8").read())["note"] == "first"
    assert json.loads(open(b, encoding="utf-8").read())["note"] == "second"


def test_disabled_logger_writes_nothing(tmp_path):
    logger = TraceLogger(log_dir=tmp_path, enabled=False)
    assert _write(logger) is None
    assert list(tmp_path.iterdir()) == []
