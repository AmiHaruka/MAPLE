import pytest

from maple.function.read.command_control import CommandControl


def test_invalid_uma_inference_logs_to_output(tmp_path):
    out = tmp_path / "bad_uma.out"

    with pytest.raises(ValueError, match="Unsupported UMA inference mode"):
        CommandControl.from_settings(
            ["#model=uma(inference=bad)", "#sp"],
            output_path=str(out),
        )

    assert "ERROR: Unsupported UMA inference mode: 'bad'." in out.read_text()


def test_uma_inference_is_normalized_in_summary(tmp_path):
    out = tmp_path / "uma.out"

    cc = CommandControl.from_settings(
        ["#model=uma(inference=Turbo)", "#sp"],
        output_path=str(out),
    )

    assert cc.params["model_options"]["inference"] == "turbo"
    assert "model          : uma(inference=turbo,size=uma-s-1p1)" in cc.summary()


def test_unknown_opt_parameter_logs_to_output(tmp_path):
    out = tmp_path / "bad_opt_param.out"

    with pytest.raises(ValueError, match="Unknown OPT parameter: 'max_itre'"):
        CommandControl.from_settings(
            ["#model=uma", "#opt(method=lbfgs,max_itre=10)"],
            output_path=str(out),
        )

    text = out.read_text()
    assert "ERROR: Unknown OPT parameter: 'max_itre'." in text
    assert "Did you mean 'max_iter'?" in text


def test_legacy_bare_opt_method_flag_is_normalized(tmp_path):
    out = tmp_path / "legacy_opt.out"

    cc = CommandControl.from_settings(
        ["#model=uma", "#opt(lbfgs)"],
        output_path=str(out),
    )

    assert cc.params["method"] == "lbfgs"
    assert "lbfgs" not in cc.params


def test_unknown_uma_model_option_logs_to_output(tmp_path):
    out = tmp_path / "bad_uma_option.out"

    with pytest.raises(ValueError, match="Unknown uma option parameter: 'infernece'"):
        CommandControl.from_settings(
            ["#model=uma(infernece=turbo)", "#sp"],
            output_path=str(out),
        )

    text = out.read_text()
    assert "ERROR: Unknown uma option parameter: 'infernece'." in text
    assert "Did you mean 'inference'?" in text


def test_unknown_md_parameter_logs_to_output(tmp_path):
    out = tmp_path / "bad_md_param.out"

    with pytest.raises(ValueError, match="Unknown MD parameter: 'stepz'"):
        CommandControl.from_settings(
            ["#model=uma", "#md(ensemble=nve,stepz=10)"],
            output_path=str(out),
        )

    assert "ERROR: Unknown MD parameter: 'stepz'." in out.read_text()
