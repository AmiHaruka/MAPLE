import torch

from maple.function.calculator.set_calculator import SetClaculator


def test_uma_turbo_cpu_fallback_logs_to_output(tmp_path):
    out = tmp_path / "uma_cpu.out"
    setter = SetClaculator(
        device=torch.device("cpu"),
        model="uma",
        output=str(out),
        model_options={"inference": "turbo"},
    )

    assert setter._coerce_uma_inference_for_device("turbo", "cpu") == "default"

    assert "UMA inference='turbo' requires CUDA" in out.read_text()
