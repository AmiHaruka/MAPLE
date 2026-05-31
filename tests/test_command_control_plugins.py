import sys
import textwrap

from maple.function.calculator.set_calculator import SetCalculator
from maple.function.read.command_control import CommandControl


def test_plugin_model_with_module_reaches_calculator_registry(tmp_path):
    plugin = tmp_path / "cc_plugin_model.py"
    plugin.write_text(
        textwrap.dedent(
            """
            from maple.function.calculator import CalcABC, register_calculator

            @register_calculator
            class PluginCalc(CalcABC):
                MODEL_NAMES = ('ccpluginmodel',)
            """
        )
    )
    sys.path.insert(0, str(tmp_path))
    try:
        cc = CommandControl.from_settings([
            "#model=ccpluginmodel(module=cc_plugin_model)",
        ])
        params = cc.as_dict()

        calc_cls = SetCalculator(
            device="cpu",
            model=params["model"],
            output="",
            model_options=params["model_options"],
        )._discover_calculator_class(params["model"])
    finally:
        sys.path.remove(str(tmp_path))

    assert params["model"] == "ccpluginmodel"
    assert params["model_options"] == {"module": "cc_plugin_model"}
    assert calc_cls.__name__ == "PluginCalc"


def test_env_plugin_model_name_is_not_rejected_by_command_control():
    cc = CommandControl.from_settings(["#model=externalmodel"])

    assert cc.as_dict()["model"] == "externalmodel"


def test_module_option_is_not_rejected_for_builtin_model():
    cc = CommandControl.from_settings(["#model=ani2x(module=my_lab.maple_plugin)"])

    assert cc.as_dict()["model_options"] == {"module": "my_lab.maple_plugin"}
