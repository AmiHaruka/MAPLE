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
