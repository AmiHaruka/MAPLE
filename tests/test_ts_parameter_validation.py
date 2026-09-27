from dataclasses import dataclass
from pathlib import Path

import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from maple.function.dispatcher.jobABC import JobABC
from maple.function.dispatcher.ts.algorithm.dimer import Dimer
from maple.function.dispatcher.ts.algorithm.neb import NEB
from maple.function.dispatcher.ts.algorithm.string import GSM
from maple.function.read.command_control import CommandControl
from maple.function.utility import Molecules


@pytest.mark.parametrize(
    ("line", "key", "expected"),
    [
        ("#opt(method=rfo,fd_batch_size=4)", "fd_batch_size", 4),
        ("#opt(method=rfo,fd_batch_size=auto)", "fd_batch_size", "auto"),
        ("#opt(method=rfo,fd_batch_size=ALL)", "fd_batch_size", "all"),
        ("#scan(method=lbfgs,scan_batch_size=3)", "scan_batch_size", 3),
        ("#scan(method=lbfgs,scan_batch_size=auto)", "scan_batch_size", "auto"),
        ("#scan(method=lbfgs,scan_batch_size=all)", "scan_batch_size", "all"),
        ("#freq(fd_batch_size=8)", "fd_batch_size", 8),
        ("#freq(fd_batch_size=auto)", "fd_batch_size", "auto"),
        ("#freq(fd_batch_size=all)", "fd_batch_size", "all"),
    ],
)
def test_public_batch_size_parameters_are_validated(line, key, expected):
    parsed = CommandControl.from_settings([line]).as_dict()
    assert parsed[key] == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            "#opt(method=rfo,fd_hessian_antisymmetry_action=warn,"
            "fd_hessian_antisymmetry_threshold=0)",
            ("warn", 0),
        ),
        (
            "#freq(fd_batch_size=all,fd_hessian_antisymmetry_action=IGNORE,"
            "fd_hessian_antisymmetry_threshold=1e-5)",
            ("ignore", 1e-5),
        ),
    ],
)
def test_fd_hessian_diagnostic_overrides_are_validated(line, expected):
    parsed = CommandControl.from_settings([line]).params
    assert parsed["fd_hessian_antisymmetry_action"] == expected[0]
    assert parsed["fd_hessian_antisymmetry_threshold"] == expected[1]


def test_frequency_fd_overrides_default_to_none():
    parsed = CommandControl.from_settings(["#freq"]).params
    assert parsed["fd_batch_size"] is None
    assert parsed["fd_hessian_antisymmetry_action"] is None
    assert parsed["fd_hessian_antisymmetry_threshold"] is None


@pytest.mark.parametrize("key", ["fd_batch_size", "scan_batch_size"])
@pytest.mark.parametrize("value", ["0", "-2", "false", "invalid"])
def test_public_batch_size_parameters_reject_invalid_values(key, value):
    task = "opt" if key == "fd_batch_size" else "scan"
    with pytest.raises(ValueError, match="positive integer"):
        CommandControl.from_settings(
            [f"#{task}(method=rfo,{key}={value})"]
        )


@pytest.mark.parametrize("value", ["0", "-2", "false", "invalid"])
def test_frequency_fd_batch_size_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        CommandControl.from_settings([f"#freq(fd_batch_size={value})"])


@pytest.mark.parametrize("task", ["opt(method=rfo", "freq("])
@pytest.mark.parametrize("value", ["error", "true", "warning"])
def test_fd_hessian_diagnostic_action_rejects_invalid_values(task, value):
    separator = "," if "method=" in task else ""
    line = (
        f"#{task}{separator}fd_hessian_antisymmetry_action={value})"
    )
    with pytest.raises(ValueError, match="must be one of"):
        CommandControl.from_settings([line])


@pytest.mark.parametrize("task", ["opt(method=rfo", "freq("])
@pytest.mark.parametrize("value", ["-1", "false", "nan", "inf", "invalid"])
def test_fd_hessian_diagnostic_threshold_rejects_invalid_values(task, value):
    separator = "," if "method=" in task else ""
    line = (
        f"#{task}{separator}fd_hessian_antisymmetry_threshold={value})"
    )
    with pytest.raises(ValueError, match="finite non-negative number"):
        CommandControl.from_settings([line])


def test_ts_parser_rejects_method_specific_typo_with_suggestion():
    with pytest.raises(ValueError, match=r"Unknown TS parameter.*Did you mean 'n_images'"):
        CommandControl.from_settings(["#ts(method=neb,n_image=4)"])


def test_ts_parser_rejects_duplicate_nested_keys():
    with pytest.raises(ValueError, match="Duplicate nested parameter: 'max_iter'"):
        CommandControl.from_settings(
            ["#ts(method=prfo,max_iter=4,max_iter=5)"]
        )


@pytest.mark.parametrize(
    ("method", "foreign_key"),
    [
        ("prfo", "n_images"),
        ("dimer", "n_images"),
        ("neb", "delta"),
        ("string", "delta"),
        ("autoneb", "delta"),
        ("dmf", "delta"),
    ],
)
def test_each_ts_method_rejects_foreign_parameters(method, foreign_key):
    with pytest.raises(ValueError, match="Unknown TS parameter"):
        CommandControl.from_settings(
            [f"#ts(method={method},{foreign_key}=1)"]
        )


def test_string_max_iter_does_not_redefine_another_optimizer_budget():
    with pytest.raises(ValueError, match="Unknown TS parameter: 'max_iter'"):
        CommandControl.from_settings(["#ts(method=string,max_iter=4)"])


def test_explicit_prfo_refinement_options_are_validated_and_preserved():
    control = CommandControl.from_settings(
        [
            "#ts(method=dmf,refine=prfo)",
            "#prfo(max_iter=7,rigid_symmetry=free_molecule)",
        ]
    )
    assert control.params["prfo"] == {
        "max_iter": 7,
        "rigid_symmetry": "free_molecule",
    }


@pytest.mark.parametrize(
    ("method", "refine", "nested_line", "refiner"),
    [
        ("neb", "nebts", "#neb(refine=nebts)", "prfo"),
        ("string", "stringts", "#string(refine=stringts)", "prfo"),
        ("dmf", "prfo", "#dmf(refine=prfo)", "prfo"),
    ],
)
def test_parser_uses_nested_effective_refine_for_refiner_options(
    method, refine, nested_line, refiner
):
    control = CommandControl.from_settings(
        [
            f"#ts(method={method})",
            nested_line,
            f"#{refiner}(max_iter=2)",
        ]
    )
    assert control.params[method]["refine"] == refine
    assert control.params[refiner]["max_iter"] == 2


@pytest.mark.parametrize(
    ("method", "nested_line"),
    [
        ("neb", "#neb(refine=bad_refine)"),
        ("string", "#string(refine=bad_refine)"),
        ("dmf", "#dmf(refine=bad_refine)"),
    ],
)
def test_parser_rejects_invalid_nested_effective_refine(method, nested_line):
    with pytest.raises(ValueError, match="Refine 'bad_refine' not implemented"):
        CommandControl.from_settings([f"#ts(method={method})", nested_line])


@pytest.mark.parametrize(
    ("method", "flat_refine", "nested_line"),
    [
        ("neb", "nebts", "#neb(refine=cineb)"),
        ("string", "stringts", "#string(refine=cistring)"),
        ("dmf", "prfo", "#dmf(refine=dimer)"),
    ],
)
def test_parser_nested_refine_override_closes_stale_refiner_options(
    method, flat_refine, nested_line
):
    with pytest.raises(ValueError, match="Unknown TS parameter: 'prfo'"):
        CommandControl.from_settings(
            [
                f"#ts(method={method},refine={flat_refine})",
                nested_line,
                "#prfo(max_iter=2)",
            ]
        )


def test_explicit_prfo_refinement_typo_is_rejected():
    with pytest.raises(ValueError, match=r"Unknown TS prfo parameter.*rigid_symmetry"):
        CommandControl.from_settings(
            [
                "#ts(method=dmf,refine=prfo)",
                "#prfo(rigid_symetry=free_molecule)",
            ]
        )


@dataclass
class _Params:
    max_iter: int = 1


class _Job(JobABC):
    def run(self):
        return None


def test_direct_api_strict_mode_rejects_unknown_keys():
    job = _Job("unused.out")
    with pytest.raises(ValueError, match=r"Unknown Synthetic parameter.*max_iter"):
        job._init_params(
            _Params,
            {"max_ite": 3},
            ("synthetic",),
            strict=True,
            context="Synthetic",
        )


def test_direct_api_nested_block_does_not_hide_top_level_typo():
    job = _Job("unused.out")
    with pytest.raises(ValueError, match="Unknown Synthetic parameter: 'max_ite'"):
        job._init_params(
            _Params,
            {"synthetic": {"max_iter": 3}, "max_ite": 4},
            ("synthetic",),
            strict=True,
            context="Synthetic",
        )


def test_direct_api_explicit_nested_block_overrides_flat_value():
    job = _Job("unused.out")
    params = job._init_params(
        _Params,
        {"synthetic": {"max_iter": 3}, "max_iter": 2},
        ("synthetic",),
        strict=True,
        context="Synthetic",
    )
    assert params.max_iter == 3


def test_direct_api_legacy_mode_remains_non_strict():
    job = _Job("unused.out")
    params = job._init_params(_Params, {"foreign": 3}, ("synthetic",))
    assert params.max_iter == 1


def test_refinement_params_use_nested_budget_and_legacy_symmetry_only():
    selected = JobABC._select_refinement_params(
        {
            "method": "string",
            "max_iter": 999,
            "rigid_symmetry": "free_molecule",
            "prfo": {"max_iter": 7},
        },
        "prfo",
    )
    assert selected == {
        "prfo": {"max_iter": 7, "rigid_symmetry": "free_molecule"}
    }


def test_parent_max_iter_is_not_forwarded_to_refinement():
    assert JobABC._select_refinement_params(
        {"method": "string", "max_iter": 999}, "prfo"
    ) == {}


def test_dimer_refinement_does_not_receive_prfo_rigid_symmetry():
    selected = JobABC._select_refinement_params(
        {
            "rigid_symmetry": "free_molecule",
            "dimer": {"max_iter": 7},
        },
        "dimer",
    )
    assert selected == {"dimer": {"max_iter": 7}}


@pytest.mark.parametrize(
    ("constructor", "context"),
    [
        (
            lambda: NEB(
                "unused.out",
                Molecules([Atoms("H"), Atoms("H")]),
                paras={"prfo": {"max_iter": 2}},
            ),
            "NEB",
        ),
        (
            lambda: GSM(
                "unused.out",
                Atoms("H"),
                Atoms("H"),
                paras={"prfo": {"max_iter": 2}},
            ),
            "String",
        ),
    ],
)
def test_direct_path_constructor_rejects_refinement_options_without_refine(
    constructor, context
):
    with pytest.raises(ValueError, match=f"Unknown {context} parameter: 'prfo'"):
        constructor()


@pytest.mark.parametrize(
    "paras",
    [
        {"dmf": {"refine": "dimer"}},
        {"rigid_symmetry": "free_molecule"},
    ],
)
def test_direct_dimer_constructor_rejects_foreign_parent_options(tmp_path, paras):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=0.0,
        forces=[[0.0, 0.0, 0.0]],
    )
    with pytest.raises(ValueError, match="Unknown Dimer parameter"):
        Dimer(
            output=str(tmp_path / "dimer.out"),
            atoms_init=atoms,
            paras=paras,
        )


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: NEB(
            "unused.out",
            Molecules([Atoms("H"), Atoms("H")]),
            paras={"refine": "nebts", "prfo": {"max_ite": 2}},
        ),
        lambda: GSM(
            "unused.out",
            Atoms("H"),
            Atoms("H"),
            paras={"refine": "stringts", "prfo": {"max_ite": 2}},
        ),
    ],
)
def test_direct_path_constructor_validates_selected_nested_refinement(constructor):
    with pytest.raises(ValueError, match=r"refinement parameter.*max_iter"):
        constructor()


@pytest.mark.parametrize(
    ("constructor", "expected_refine"),
    [
        (
            lambda: NEB(
                "unused.out",
                Molecules([Atoms("H"), Atoms("H")]),
                paras={"refine": "nebts", "prfo": {"max_iter": 2}},
            ),
            "nebts",
        ),
        (
            lambda: GSM(
                "unused.out",
                Atoms("H"),
                Atoms("H"),
                paras={"refine": "stringts", "prfo": {"max_iter": 2}},
            ),
            "stringts",
        ),
    ],
)
def test_direct_path_constructor_accepts_valid_selected_refinement(
    constructor, expected_refine
):
    assert constructor().params.refine == expected_refine


def test_direct_neb_merges_flat_refine_with_nested_method_options():
    job = NEB(
        "unused.out",
        Molecules([Atoms("H"), Atoms("H")]),
        paras={
            "refine": "nebts",
            "prfo": {"max_iter": 7},
            "neb": {"n_images": 3},
        },
    )
    assert job.params.refine == "nebts"
    assert job.params.n_images == 5
    assert JobABC._select_refinement_params(
        job._refinement_paras,
        "prfo",
    ) == {"prfo": {"max_iter": 7}}


def test_direct_nested_ts_refinement_is_validated_and_forwarded():
    job = GSM(
        "unused.out",
        Atoms("H"),
        Atoms("H"),
        paras={
            "ts": {
                "refine": "stringts",
                "prfo": {"max_iter": 7},
            }
        },
    )
    assert job.params.refine == "stringts"
    assert JobABC._select_refinement_params(
        job._refinement_paras,
        "prfo",
    ) == {"prfo": {"max_iter": 7}}


def test_direct_nested_ts_refinement_rejects_nested_typo():
    with pytest.raises(ValueError, match=r"refinement parameter.*max_iter"):
        GSM(
            "unused.out",
            Atoms("H"),
            Atoms("H"),
            paras={
                "ts": {
                    "refine": "stringts",
                    "prfo": {"max_ite": 7},
                }
            },
        )


def test_dmf_refinement_language_is_candidate_only():
    source = Path(
        "maple/function/dispatcher/ts/algorithm/dmf.py"
    ).read_text(encoding="utf-8")
    assert "Geometry-refined candidate; not frequency/IRC-verified." in source
    assert "REFINED CANDIDATE STRUCTURE" in source
    assert "true first-order saddle" not in source
    assert "Energy (refined TS)" not in source
    assert "REFINED TS STRUCTURE" not in source
