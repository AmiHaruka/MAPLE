from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-maple")

import torch


ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    example_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run parmfit correction and draw per-bond MLP/pre-mSem/post-mSem/refit-MM plots.",
    )
    parser.add_argument("--mol2", type=Path, default=example_dir / "demo.mol2")
    parser.add_argument("--ff-mol2", type=Path, default=example_dir / "demo_ff.mol2")
    parser.add_argument("--frcmod", type=Path, default=example_dir / "demo_ff.frcmod")
    parser.add_argument("--model", default="ani2x", help="MAPLE model name.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cpu or cuda:0")
    parser.add_argument("--charge", type=int, default=0, help="Total charge. Default: 0")
    parser.add_argument("--mult", type=int, default=1, help="Spin multiplicity. Default: 1")
    parser.add_argument(
        "--torsion-bonds",
        default=None,
        help="Optional comma-separated 1-based center bonds, e.g. '8-15,3-7'. Default: auto-select.",
    )
    parser.add_argument("--scan-step", type=float, default=10.0, help="Torsion scan step in degrees.")
    parser.add_argument("--scan-steps", type=int, default=36, help="Number of torsion scan steps.")
    parser.add_argument("--refine-rounds", type=int, default=10, help="Fixed-scan refinement rounds.")
    parser.add_argument("--refine-max-iter", type=int, default=50, help="Per-bond refinement max iterations.")
    parser.add_argument("--refine-tol", type=float, default=1.0e-4, help="Refinement acceptance tolerance in kcal/mol.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Main correction output path. Default: <mol2_stem>_torsionfit_v2.out",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for PNG plots. Default: alongside the output file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from ase import Atoms  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("This script requires ASE to build the molecular structure.") from exc

    try:
        import matplotlib.pyplot as plt  # noqa: F401
        from PIL import Image  # noqa: F401
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem import AllChem  # noqa: F401
        from rdkit.Chem.Draw import rdMolDraw2D  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script requires matplotlib, Pillow, and RDKit for torsion-fit visualization."
        ) from exc

    from maple.function.calculator.set_calculator import SetClaculator
    from maple.function.dispatcher.parmfit.correction.correction import Correction
    from maple.function.dispatcher.parmfit.utils.mSeminario import apply_mseminario
    from maple.function.dispatcher.parmfit.utils.mmcalc import build_mm_topology_cache, evaluate_mm_energy
    from maple.function.dispatcher.parmfit.utils.readparm import build_correction_parameter_set
    from maple.function.dispatcher.parmfit.utils.torsionfit import (
        center_bond_dihedrals,
        evaluate_refit_objective,
        read_scan_xyz,
    )

    from maple.function.dispatcher.parmfit.example.test_torsionfit_viz import (
        apply_default_thresholds,
        load_mol2_atoms,
        load_rdkit_mol,
        render_highlighted_molecule,
    )

    mol2_path = args.mol2.resolve()
    ff_mol2_path = args.ff_mol2.resolve()
    frcmod_path = args.frcmod.resolve()
    for path in (mol2_path, ff_mol2_path, frcmod_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device '{args.device}' but CUDA is not available.")

    base = mol2_path.with_suffix("")
    output_path = args.output.resolve() if args.output is not None else base.with_name(base.name + "_torsionfit_v2.out")
    plot_dir = args.plot_dir.resolve() if args.plot_dir is not None else output_path.parent
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    atoms = load_mol2_atoms(mol2_path)
    atoms.info["charge"] = args.charge
    atoms.info["mult"] = args.mult
    apply_default_thresholds(atoms)
    atoms.calc = SetClaculator(
        device=torch.device(args.device),
        model=args.model,
        output=str(output_path),
        atoms=atoms,
        d4=False,
        implicit="None",
        solvent="None",
    ).set_calculator()

    # Snapshot 1: original MM parameter set before mSeminario.
    pre_msem_parameter_set = build_correction_parameter_set(atoms, str(ff_mol2_path), str(frcmod_path))

    correction = Correction(
        output=str(output_path),
        atoms=atoms,
        params={
            "mol2": str(ff_mol2_path),
            "frcmod": str(frcmod_path),
            "torsionfit": True,
            "torsion_bonds": args.torsion_bonds,
            "torsion_scan_step": args.scan_step,
            "torsion_scan_steps": args.scan_steps,
            "torsion_refine_rounds": args.refine_rounds,
            "torsion_refine_max_iter": args.refine_max_iter,
            "torsion_refine_tol": args.refine_tol,
        },
    )
    correction.run()

    if not correction.torsion_fit_reports or correction.result is None:
        print("No torsion fits were executed. Check the main output for selection details.")
        print(f"Log written to {output_path}")
        return 0

    # Snapshot 2: post-mSem, pre-torsion-fit parameter set reconstructed on optimized geometry.
    post_msem_parameter_set = build_correction_parameter_set(correction.atoms, str(ff_mol2_path), str(frcmod_path))
    hessian = correction._get_hessian_cart()
    scaling = float(correction.params.vibrational_scaling)
    apply_mseminario(correction.atoms, hessian, post_msem_parameter_set.bonds, post_msem_parameter_set.angles, scaling)

    rdkit_mol = load_rdkit_mol(mol2_path)
    written = []
    for stage1_result in correction.torsion_fit_reports:
        center_bond = stage1_result["center_bond"]
        png_path = plot_dir / f"{output_path.stem}_torsionfit_{center_bond[0]}-{center_bond[1]}.png"

        scan_data = correction.torsion_scan_data.get(center_bond)
        if scan_data is None and stage1_result["scan_source_path"] is not None:
            scan_data = read_scan_xyz(stage1_result["scan_source_path"])
        if scan_data is None:
            raise ValueError(f"Missing scan data for center bond {center_bond}.")

        pre_msem_rel = relative_mm_curve(scan_data, pre_msem_parameter_set)
        post_msem_rel = relative_mm_curve(scan_data, post_msem_parameter_set)

        final_dihedrals = center_bond_dihedrals(correction.result, center_bond)
        final_terms = [
            [type(term)(term.kPhi, term.period, term.phase) for term in dihedral.terms]
            for dihedral in final_dihedrals
        ]
        _, final_refit_rel, _ = evaluate_refit_objective(
            scan_data,
            correction.result,
            center_bond,
            final_terms,
        )

        qm_rel = qm_relative_curve(stage1_result)
        rmse_pre, _ = rmse_vs_mlp(qm_rel, pre_msem_rel)
        rmse_post, _ = rmse_vs_mlp(qm_rel, post_msem_rel)
        rmse_refit, _ = rmse_vs_mlp(qm_rel, final_refit_rel)

        draw_torsion_fit_figure_v2(
            rdkit_mol=rdkit_mol,
            result=stage1_result,
            output_path=png_path,
            pre_msem_rel=pre_msem_rel,
            post_msem_rel=post_msem_rel,
            final_refit_rel=final_refit_rel,
            rmse_pre=rmse_pre,
            rmse_post=rmse_post,
            rmse_refit=rmse_refit,
        )
        written.append(png_path)
        print(
            f"Center bond {center_bond}: "
            f"pre-mSem RMSE={rmse_pre:.6f}  "
            f"post-mSem RMSE={rmse_post:.6f}  "
            f"final refit RMSE={rmse_refit:.6f} kcal/mol"
        )

    print(f"Correction + torsion fitting complete. Log written to {output_path}")
    for png in written:
        print(f"Wrote {png}")
    return 0


def relative_mm_curve(scan_data, parameter_set) -> np.ndarray:
    from maple.function.dispatcher.parmfit.utils.mmcalc import build_mm_topology_cache, evaluate_mm_energy

    cache = build_mm_topology_cache(parameter_set)
    total = np.asarray(
        [evaluate_mm_energy(atoms, parameter_set, topology_cache=cache).total for atoms in scan_data["frames"]],
        dtype=float,
    )
    return total - np.min(total)


def qm_relative_curve(result) -> np.ndarray:
    return np.asarray(result["qm_rel"], dtype=float)


def rmse_vs_mlp(qm_rel, mm_rel) -> tuple[float, np.ndarray]:
    qm_rel = np.asarray(qm_rel, dtype=float)
    mm_rel = np.asarray(mm_rel, dtype=float)
    residual = mm_rel - qm_rel
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    return rmse, residual

def render_highlighted_molecule(rdkit_mol, result):
    from io import BytesIO
    from PIL import Image
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.Mol(rdkit_mol)
    group_atoms = sorted({atom - 1 for dihedral in result["target_dihedrals"] for atom in dihedral.atoms})
    center_atoms = [result["center_bond"][0] - 1, result["center_bond"][1] - 1]

    atom_colors = {atom: (0.96, 0.55, 0.55) for atom in group_atoms}
    atom_radii = {atom: 0.36 for atom in group_atoms}
    for atom in center_atoms:
        atom_colors[atom] = (0.88, 0.12, 0.12)
        atom_radii[atom] = 0.46

    bond = mol.GetBondBetweenAtoms(center_atoms[0], center_atoms[1])
    highlight_bonds = [bond.GetIdx()] if bond is not None else []
    bond_colors = {bond_idx: (0.88, 0.12, 0.12) for bond_idx in highlight_bonds}

    drawer = rdMolDraw2D.MolDraw2DCairo(460, 420)
    opts = drawer.drawOptions()
    opts.padding = 0.05
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=group_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText()))


def draw_torsion_fit_figure_v2(
    rdkit_mol,
    result,
    output_path: Path,
    pre_msem_rel,
    post_msem_rel,
    final_refit_rel,
    rmse_pre: float,
    rmse_post: float,
    rmse_refit: float,
) -> None:
    import matplotlib.pyplot as plt

    angles = list(np.asarray(result["angles_deg"], dtype=float))
    qm = list(np.asarray(result["qm_rel"], dtype=float))

    fig, (ax_curve, ax_mol) = plt.subplots(
        1,
        2,
        figsize=(12.2, 4.8),
        gridspec_kw={"width_ratios": [2.7, 1.25]},
    )

    ax_curve.plot(angles, qm, marker="o", lw=1.9, ms=4.8, label="MLP", color="#2a6fdb")
    ax_curve.plot(
        angles,
        pre_msem_rel,
        marker="s",
        lw=1.6,
        ms=4.0,
        label=f"pre-mSem MM (RMSE={rmse_pre:.3f})",
        color="#c0392b",
    )
    ax_curve.plot(
        angles,
        post_msem_rel,
        marker="D",
        lw=1.5,
        ms=3.8,
        label=f"post-mSem MM (RMSE={rmse_post:.3f})",
        color="#f39c12",
    )
    ax_curve.plot(
        angles,
        final_refit_rel,
        marker="^",
        lw=1.7,
        ms=4.2,
        label=f"final refit MM (RMSE={rmse_refit:.3f})",
        color="#148f77",
    )
    ax_curve.set_xlabel("Dihedral Angle [degrees]")
    ax_curve.set_ylabel("Relative Energy [kcal/mol]")
    ax_curve.set_title(f"Center Bond {result['center_bond'][0]}-{result['center_bond'][1]}")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(frameon=False, loc="best")

    image = render_highlighted_molecule(rdkit_mol, result)
    ax_mol.imshow(image)
    ax_mol.axis("off")
    ax_mol.set_title("Torsion Group")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
