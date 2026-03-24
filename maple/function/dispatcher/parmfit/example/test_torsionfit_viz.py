from __future__ import annotations

import argparse
from io import BytesIO
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-maple")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_THRESHOLDS = {
    "f_max_th": 0.00285,
    "f_rms_th": 0.00190,
    "dp_max_th": 0.00315,
    "dp_rms_th": 0.00210,
}


def parse_args() -> argparse.Namespace:
    example_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run parmfit correction + torsion fitting and draw per-bond MLP/orig-MM/refit-MM plots.",
    )
    parser.add_argument(
        "--mol2",
        type=Path,
        default=example_dir / "demo.mol2",
        help="Coordinate mol2 file used to build the starting Atoms geometry.",
    )
    parser.add_argument(
        "--ff-mol2",
        type=Path,
        default=example_dir / "demo_ff.mol2",
        help="Force-field mol2 used by correction parmfit.",
    )
    parser.add_argument(
        "--frcmod",
        type=Path,
        default=example_dir / "demo_ff.frcmod",
        help="frcmod file used by correction parmfit.",
    )
    parser.add_argument("--model", default="ani2x", help="MAPLE model name.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0")
    parser.add_argument("--charge", type=int, default=0, help="Total charge. Default: 0")
    parser.add_argument("--mult", type=int, default=1, help="Spin multiplicity. Default: 1")
    parser.add_argument(
        "--torsion-bonds",
        default=None,
        help="Optional comma-separated 1-based center bonds, e.g. '8-15,3-7'. Default: auto-select.",
    )
    parser.add_argument("--scan-step", type=float, default=5.0, help="Torsion scan step in degrees.")
    parser.add_argument("--scan-steps", type=int, default=72, help="Number of torsion scan steps.")
    parser.add_argument(
        "--refine-rounds",
        type=int,
        default=50,
        help="Optional fixed-scan torsion refinement rounds. Default: use Correction default.",
    )
    parser.add_argument(
        "--refine-max-iter",
        type=int,
        default=50,
        help="Optional per-bond max iterations for fixed-scan refinement. Default: use Correction default.",
    )
    parser.add_argument(
        "--refine-tol",
        type=float,
        default=1e-4,
        help="Optional acceptance tolerance for fixed-scan refinement in kcal/mol. Default: use Correction default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Main correction output path. Default: <mol2_stem>_torsionfit.out",
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
    from maple.function.dispatcher.parmfit.utils.torsionfit import (
        center_bond_dihedrals,
        evaluate_refit_objective,
        read_scan_xyz,
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
    output_path = args.output.resolve() if args.output is not None else base.with_name(base.name + "_torsionfit.out")
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

    correction_params = {
        "mol2": str(ff_mol2_path),
        "frcmod": str(frcmod_path),
        "torsionfit": True,
        "torsion_bonds": args.torsion_bonds,
        "torsion_scan_step": args.scan_step,
        "torsion_scan_steps": args.scan_steps,
    }
    if args.refine_rounds is not None:
        correction_params["torsion_refine_rounds"] = args.refine_rounds
    if args.refine_max_iter is not None:
        correction_params["torsion_refine_max_iter"] = args.refine_max_iter
    if args.refine_tol is not None:
        correction_params["torsion_refine_tol"] = args.refine_tol

    correction = Correction(
        output=str(output_path),
        atoms=atoms,
        params=correction_params,
    )
    correction.run()

    if not correction.torsion_fit_reports:
        print("No torsion fits were executed. Check the main output for selection details.")
        print(f"Log written to {output_path}")
        return 0

    rdkit_mol = load_rdkit_mol(mol2_path)
    written = []
    for result in correction.torsion_fit_reports:
        center_bond = result["center_bond"]
        png_path = plot_dir / f"{output_path.stem}_torsionfit_{center_bond[0]}-{center_bond[1]}.png"
        final_refit = None
        if correction.result is not None:
            scan_data = correction.torsion_scan_data.get(center_bond)
            if scan_data is None and result["scan_source_path"] is not None:
                scan_data = read_scan_xyz(result["scan_source_path"])
            if scan_data is not None:
                final_dihedrals = center_bond_dihedrals(correction.result, center_bond)
                final_terms = [
                    [type(term)(term.kPhi, term.period, term.phase) for term in dihedral.terms]
                    for dihedral in final_dihedrals
                ]
                _, final_refit, _ = evaluate_refit_objective(
                    scan_data,
                    correction.result,
                    center_bond,
                    final_terms,
                )
        draw_torsion_fit_figure(rdkit_mol, result, png_path, final_refit_rel=final_refit)
        written.append(png_path)

    print(f"Correction + torsion fitting complete. Log written to {output_path}")
    for png in written:
        print(f"Wrote {png}")
    return 0


def load_mol2_atoms(path: Path):
    from ase import Atoms

    symbols: list[str] = []
    positions: list[list[float]] = []
    charges: list[float] = []
    in_atom_section = False

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("@<TRIPOS>"):
                in_atom_section = stripped.upper() == "@<TRIPOS>ATOM"
                continue
            if not in_atom_section:
                continue

            parts = raw.split()
            if len(parts) < 6:
                raise ValueError(f"Invalid mol2 atom line in {path}: {raw.rstrip()}")
            atom_name = parts[1]
            x, y, z = map(float, parts[2:5])
            atom_type = parts[5]
            charge = float(parts[8]) if len(parts) >= 9 else 0.0

            symbols.append(guess_symbol(atom_name, atom_type))
            positions.append([x, y, z])
            charges.append(charge)

    if not symbols:
        raise ValueError(f"mol2 file has no @<TRIPOS>ATOM section with coordinates: {path}")

    atoms = Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float))
    if hasattr(atoms, "set_initial_charges"):
        atoms.set_initial_charges(np.asarray(charges, dtype=float))
    return atoms


def guess_symbol(atom_name: str, atom_type: str) -> str:
    from ase.data import chemical_symbols

    allowed = sorted((symbol for symbol in chemical_symbols if symbol), key=len, reverse=True)
    candidates = [atom_type.split(".", 1)[0], atom_name]
    for raw in candidates:
        letters = re.sub(r"[^A-Za-z]", "", raw)
        if not letters:
            continue
        normalized = letters[0].upper() + letters[1:].lower()
        for symbol in allowed:
            if normalized.startswith(symbol):
                return symbol
        single = normalized[0]
        if single in allowed:
            return single
    raise ValueError(f"Failed to infer element symbol from mol2 atom '{atom_name}' / '{atom_type}'.")


def apply_default_thresholds(atoms) -> None:
    for key, value in DEFAULT_THRESHOLDS.items():
        setattr(atoms, key, value)


def load_rdkit_mol(path: Path):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromMol2File(str(path), sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError(f"RDKit failed to read mol2 file: {path}")
    AllChem.Compute2DCoords(mol)
    return mol


def draw_torsion_fit_figure(rdkit_mol, result, output_path: Path, final_refit_rel=None) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image
    from rdkit.Chem.Draw import rdMolDraw2D

    angles = list(np.asarray(result["angles_deg"], dtype=float))
    qm = list(np.asarray(result["qm_rel"], dtype=float))
    orig = list(np.asarray(result["orig_mm_rel"], dtype=float))
    refit = (
        list(np.asarray(final_refit_rel, dtype=float))
        if final_refit_rel is not None
        else list(np.asarray(result["mm_refit_rel"], dtype=float))
    )

    fig, (ax_curve, ax_mol) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.6),
        gridspec_kw={"width_ratios": [2.5, 1.25]},
    )

    ax_curve.plot(angles, qm, marker="o", lw=1.8, ms=4.5, label="MLP", color="#2a6fdb")
    ax_curve.plot(angles, orig, marker="s", lw=1.6, ms=4.0, label="orig MM", color="#d94841")
    ax_curve.plot(angles, refit, marker="^", lw=1.6, ms=4.0, label="refit MM", color="#1f9d8f")
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


def render_highlighted_molecule(rdkit_mol, result):
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


if __name__ == "__main__":
    raise SystemExit(main())
