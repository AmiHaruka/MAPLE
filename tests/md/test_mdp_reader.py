import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from maple.function.dispatcher.md.mdp_reader import parse_mdp, _coerce


def test_coerce_types():
    assert _coerce('42') == 42
    assert isinstance(_coerce('42'), int)
    assert _coerce('3.14') == pytest.approx(3.14)
    assert isinstance(_coerce('3.14'), float)
    assert _coerce('yes') is True
    assert _coerce('no') is False
    assert _coerce('true') is True
    assert _coerce('false') is False
    assert _coerce('langevin') == 'langevin'


def test_parse_mdp_basic(tmp_path):
    mdp = tmp_path / "run.mdp"
    mdp.write_text("""\
; NVT production run
ensemble = nvt
timestep = 1.0   ; fs
steps    = 100000
temperature = 300.0
thermostat = langevin
""")
    result = parse_mdp(str(mdp))
    assert result['ensemble'] == 'nvt'
    assert result['timestep'] == pytest.approx(1.0)
    assert result['steps'] == 100000
    assert result['temperature'] == pytest.approx(300.0)
    assert result['thermostat'] == 'langevin'


def test_parse_mdp_hash_comments(tmp_path):
    mdp = tmp_path / "run.mdp"
    mdp.write_text("key = value  # this is a comment\n")
    result = parse_mdp(str(mdp))
    assert result['key'] == 'value'


def test_parse_mdp_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_mdp("/nonexistent/path/run.mdp")


def test_parse_mdp_bad_line(tmp_path):
    mdp = tmp_path / "bad.mdp"
    mdp.write_text("this line has no equals sign\n")
    with pytest.raises(ValueError, match="expected 'key = value'"):
        parse_mdp(str(mdp))


def test_mdp_loaded_via_command_control(tmp_path):
    """MDP file params are merged when #md(mdp=...) is specified."""
    from maple.function.read.command_control import CommandControl
    mdp = tmp_path / "run.mdp"
    mdp.write_text("timestep = 2.0\ntemperature = 400.0\n")
    lines = [f"#md(ensemble=nvt, mdp={mdp})"]
    cc = CommandControl.from_settings(lines)
    # ensemble was specified inline -> should be nvt
    assert cc.params['ensemble'] == 'nvt'
    # timestep was in MDP and not overridden inline -> should be 2.0
    assert cc.params['timestep'] == pytest.approx(2.0)


def test_inline_overrides_mdp(tmp_path):
    """Inline params take precedence over MDP file values."""
    from maple.function.read.command_control import CommandControl
    mdp = tmp_path / "run.mdp"
    mdp.write_text("timestep = 2.0\n")
    # inline specifies timestep=0.5 which should win
    lines = [f"#md(ensemble=nve, mdp={mdp}, timestep=0.5)"]
    cc = CommandControl.from_settings(lines)
    assert cc.params['timestep'] == pytest.approx(0.5)


def test_inline_default_value_still_overrides_mdp(tmp_path):
    """Explicit inline values must win even when equal to the built-in default."""
    from maple.function.read.command_control import CommandControl
    mdp = tmp_path / "run.mdp"
    mdp.write_text("temperature = 400.0\n")
    lines = [f"#md(ensemble=nve, mdp={mdp}, temperature=300.0)"]
    cc = CommandControl.from_settings(lines)
    assert cc.params['temperature'] == pytest.approx(300.0)


def test_parse_mdp_rejects_empty_key(tmp_path):
    mdp = tmp_path / "bad_key.mdp"
    mdp.write_text("= 1\n")
    with pytest.raises(ValueError, match="empty key"):
        parse_mdp(str(mdp))


def test_parse_mdp_rejects_empty_value(tmp_path):
    mdp = tmp_path / "bad_value.mdp"
    mdp.write_text("timestep =\n")
    with pytest.raises(ValueError, match="empty value"):
        parse_mdp(str(mdp))


def test_mdp_restart_and_rst_every_are_loaded(tmp_path):
    from maple.function.read.command_control import CommandControl

    mdp = tmp_path / "run.mdp"
    mdp.write_text("restart = true\nrst_every = 250\n")
    cc = CommandControl.from_settings([f"#md(ensemble=nvt, mdp={mdp})"])

    assert cc.params["restart"] is True
    assert cc.params["rst_every"] == 250


def test_md_defaults_use_restart_not_resume(tmp_path):
    from maple.function.read.command_control import CommandControl

    # Create a minimal MDP file
    mdp = tmp_path / "test.mdp"
    mdp.write_text("integrator = md\n")

    cc = CommandControl.from_settings([f"#md(mdp={mdp})"])

    assert "restart" in cc.params
    assert cc.params["restart"] is False
    assert cc.params["rst_every"] == 1000
    assert "resume" not in cc.params
    assert "init_from" not in cc.params
