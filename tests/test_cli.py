"""CLI and logging setup.

Thin surfaces, but both are how an operator interacts with the pipeline, and a
config command that crashes is worse than no config command.
"""

from __future__ import annotations

import json

import pytest
from structlog.testing import capture_logs

from born_field.cli import main
from born_field.logging import configure_logging, get_logger


def test_config_json_is_parseable(capsys):
    assert main(["config", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["grid"]["cell_size_m"] > 0
    assert "true_alpha" in payload["generator"]


def test_logs_go_to_stderr_not_stdout(capsys):
    """Stdout is a data channel; polluting it breaks ``--json | jq``."""
    assert main(["config"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "effective_config" in captured.err


def test_unknown_command_exits_nonzero():
    with pytest.raises(SystemExit):
        main(["nonsense"])


def test_settings_read_from_environment(monkeypatch, capsys):
    """Nested settings must be overridable without editing code."""
    monkeypatch.setenv("BORNFIELD_GENERATOR__TRUE_ALPHA", "1.9")
    assert main(["config", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["generator"]["true_alpha"] == 1.9


def test_configure_logging_is_idempotent():
    configure_logging(level="INFO")
    configure_logging(level="DEBUG")  # must not raise on repeat configuration
    assert get_logger("test", cell_id="abc") is not None


def test_logger_emits_bound_context():
    """Context bound at construction must survive to the emitted event.

    This is what makes "why did this cell score high on Tuesday" answerable
    from logs rather than by re-running the model.
    """
    with capture_logs() as events:
        get_logger("test", cell_id="abc", model_version="0.1.0").info("scored")

    assert events[0]["cell_id"] == "abc"
    assert events[0]["model_version"] == "0.1.0"
    assert events[0]["event"] == "scored"
