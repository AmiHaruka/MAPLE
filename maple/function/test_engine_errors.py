from maple.function.engine import engine


def test_engine_error_fallback_logs_unhandled_exception(tmp_path):
    out = tmp_path / "engine.out"

    engine._append_error_if_missing(str(out), RuntimeError("boom"))

    assert "ERROR: RuntimeError: boom" in out.read_text()
