import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers, vdw_radii


DATA_DIR = Path(__file__).with_name("data")
AVOGADRO = 6.02214076e23
ANGSTROM3_PER_ML = 1.0e24
WATER_MOLAR_MASS_G_MOL = 18.01528
DEFAULT_WATER_DENSITY_G_ML = 1.0
DEFAULT_VDW_SCALE = 0.57
DEFAULT_VDW_FALLBACK_RADIUS = 1.05


@dataclass(frozen=True)
class SolventTemplate:
    coords: np.ndarray
    symbols: np.ndarray
    residue_ids: np.ndarray
    groups: list[list[int]]
    cryst1_cellpar: Optional[tuple[float, float, float, float, float, float]] = None


def _element_from_pdb_line(line: str) -> str:
    elem = line[76:78].strip() if len(line) >= 78 else ""
    if elem:
        return elem[0].upper() + elem[1:].lower()

    atom_name = line[12:16].strip()
    letters = "".join(ch for ch in atom_name if ch.isalpha())
    if not letters:
        return atom_name[:1].upper()
    if len(letters) >= 2 and letters[:2].upper() in {
        "CL", "BR", "NA", "MG", "AL", "SI", "CA", "FE", "ZN", "CU",
        "MN", "CO", "NI", "LI", "BE", "NE", "AR", "KR", "XE", "HE",
    }:
        return letters[0].upper() + letters[1].lower()
    return letters[0].upper()


def _parse_atom_coords(line: str) -> tuple[float, float, float]:
    try:
        return float(line[30:38]), float(line[38:46]), float(line[46:54])
    except ValueError:
        parts = line.split()
        if len(parts) < 8:
            raise
        return float(parts[5]), float(parts[6]), float(parts[7])


def _parse_residue_key(line: str, ter_index: int) -> tuple:
    resname = line[17:20].strip() if len(line) >= 20 else ""
    chain_id = line[21:22].strip() if len(line) >= 22 else ""
    resseq = line[22:26].strip() if len(line) >= 26 else ""
    insertion_code = line[26:27].strip() if len(line) >= 27 else ""
    if resname or chain_id or resseq or insertion_code:
        return (chain_id, resseq, insertion_code, resname)
    return ("TER", ter_index)


def _parse_cryst1(line: str) -> Optional[tuple[float, float, float, float, float, float]]:
    parts = line.split()
    if len(parts) < 7:
        return None
    try:
        a, b, c = (float(parts[i]) for i in range(1, 4))
        alpha, beta, gamma = (float(parts[i]) for i in range(4, 7))
    except ValueError:
        return None
    return (a, b, c, alpha, beta, gamma)


def parse_pdb_template(pdbfile: str | os.PathLike[str]) -> SolventTemplate:
    """Parse a pure-solvent PDB template, grouping one molecule per residue."""
    coords: list[list[float]] = []
    symbols: list[str] = []
    residue_ids: list[int] = []
    groups: list[list[int]] = []
    residue_to_group: dict[tuple, int] = {}
    cryst1_cellpar = None
    ter_index = 0

    with open(pdbfile, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("CRYST1"):
                cryst1_cellpar = _parse_cryst1(line)
                continue
            if line.startswith(("ATOM", "HETATM")):
                key = _parse_residue_key(line, ter_index)
                group_id = residue_to_group.get(key)
                if group_id is None:
                    group_id = len(groups)
                    residue_to_group[key] = group_id
                    groups.append([])

                coords.append(list(_parse_atom_coords(line)))
                symbols.append(_element_from_pdb_line(line))
                atom_index = len(coords) - 1
                groups[group_id].append(atom_index)
                residue_ids.append(group_id)
                continue
            if line.startswith("TER"):
                ter_index += 1

    if not coords:
        raise ValueError(f"No ATOM/HETATM records found in solvent template: {pdbfile}")

    return SolventTemplate(
        coords=np.asarray(coords, dtype=np.float64),
        symbols=np.asarray(symbols, dtype=object),
        residue_ids=np.asarray(residue_ids, dtype=np.int64),
        groups=groups,
        cryst1_cellpar=cryst1_cellpar,
    )


def parse_pdb_residue_groups(pdbfile):
    """Backward-compatible parser returning coords, symbols, and molecule groups."""
    template = parse_pdb_template(pdbfile)
    return template.coords, template.symbols, template.groups


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    raise ValueError(f"Expected boolean value, got {value!r}.")


def _uniform_quaternion_rotation(rng: np.random.Generator) -> np.ndarray:
    """Return a uniform 3D rotation matrix sampled through a unit quaternion."""
    u1, u2, u3 = rng.random(3)
    qx = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    qy = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    qz = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    qw = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    if len(coords) < 2:
        return np.empty(0, dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    upper = np.triu_indices(len(coords), k=1)
    return np.linalg.norm(diff[upper], axis=1)


def _molecular_weight(symbols: Sequence[str]) -> float:
    mass = 0.0
    for symbol in symbols:
        atomic_number = atomic_numbers.get(str(symbol), 0)
        if atomic_number <= 0:
            raise ValueError(f"Unknown element symbol in solvent template: {symbol!r}")
        mass += float(atomic_masses[atomic_number])
    return mass


def _molecule_number_density(density_g_ml: float, molar_mass_g_mol: float) -> float:
    return density_g_ml / molar_mass_g_mol * AVOGADRO / ANGSTROM3_PER_ML


def _default_clash_method(params: dict) -> str:
    method = params.get("clash_method")
    if method is not None:
        return str(method).lower()
    if "tolerance" in params or "clash_cutoff" in params:
        return "distance"
    return "vdw"


def _element_vdw_radius(symbol: str, fallback_radius: float) -> float:
    atomic_number = atomic_numbers.get(str(symbol), 0)
    if atomic_number <= 0:
        return fallback_radius
    radius = float(vdw_radii[atomic_number])
    if not math.isfinite(radius) or radius <= 0:
        return fallback_radius
    return radius


def _element_vdw_radii(symbols: Sequence[str], fallback_radius: float) -> np.ndarray:
    return np.asarray(
        [_element_vdw_radius(symbol, fallback_radius) for symbol in symbols],
        dtype=np.float64,
    )


def _has_scaled_vdw_clash(
    solvent_coords: np.ndarray,
    solvent_symbols: Sequence[str],
    solute_coords: np.ndarray,
    solute_symbols: Sequence[str],
    scale: float,
    fallback_radius: float,
) -> bool:
    solvent_radii = _element_vdw_radii(solvent_symbols, fallback_radius)
    solute_radii = _element_vdw_radii(solute_symbols, fallback_radius)
    distances = np.linalg.norm(
        solvent_coords[:, None, :] - solute_coords[None, :, :],
        axis=-1,
    )
    thresholds = (solvent_radii[:, None] + solute_radii[None, :]) * scale
    return bool(np.any(distances < thresholds))


def _write_xyz(path: Path, atoms: Atoms, comment: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(atoms)}\n")
        handle.write(f"{comment}\n")
        for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            handle.write(f"{symbol:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n")


def _write_pdb(path: Path, atoms: Atoms, solute_count: int, molecule_ids: np.ndarray) -> None:
    solvent_residue_numbers: dict[int, int] = {}
    next_residue = 2
    with path.open("w", encoding="utf-8") as handle:
        for idx, (symbol, (x, y, z), molecule_id) in enumerate(
            zip(atoms.get_chemical_symbols(), atoms.get_positions(), molecule_ids),
            start=1,
        ):
            if idx <= solute_count:
                resname = "SOL"
                resseq = 1
                atom_name = symbol[:2].upper()
            else:
                resname = "WAT" if symbol in {"O", "H"} else "SLV"
                mid = int(molecule_id)
                if mid not in solvent_residue_numbers:
                    solvent_residue_numbers[mid] = next_residue
                    next_residue += 1
                resseq = solvent_residue_numbers[mid]
                atom_name = symbol[:2].upper()
            handle.write(
                f"HETATM{idx:5d} {atom_name:<4s} {resname:>3s} A{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {symbol:>2s}\n"
            )
        handle.write("END\n")


class ExplicitSolv():
    def __new__(cls, atoms: Atoms, params: dict, device: torch.device, output: str):
        obj = super().__new__(cls)
        obj.device = device
        obj.output = output
        obj.atoms = atoms.copy()
        obj.params = dict(params or {})
        obj.solute_count = len(atoms)

        obj.solv_name = str(obj.params.get("explicit", "water")).lower()
        obj.shape = str(obj.params.get("shape", "sphere")).lower()
        if obj.shape == "box":
            obj.shape = "cube"
        obj.radius = float(obj.params.get("radius", 10.0))
        obj.box_size = obj.params.get("box_size")
        obj.density = float(obj.params.get("density", DEFAULT_WATER_DENSITY_G_ML))
        obj.density_scale = float(obj.params.get("density_scale", 1.0))
        obj.number = obj.params.get("number")
        obj.clash_method = _default_clash_method(obj.params)
        obj.tolerance = float(obj.params.get("tolerance", obj.params.get("clash_cutoff", 2.0)))
        obj.vdw_scale = float(obj.params.get("vdw_scale", DEFAULT_VDW_SCALE))
        obj.vdw_fallback_radius = float(
            obj.params.get("vdw_fallback_radius", DEFAULT_VDW_FALLBACK_RADIUS)
        )
        obj.randomize = _as_bool(obj.params.get("randomize"), True)
        obj.write_shell = _as_bool(obj.params.get("write_shell"), False)
        obj.shell_cutoff = obj.params.get("shell_cutoff")
        obj.seed = obj.params.get("seed", 0)
        obj.rng = np.random.default_rng(None if obj.seed == -1 else obj.seed)

        obj._validate_options()

        obj.data_path = DATA_DIR / f"{obj.solv_name}.pdb"
        if not obj.data_path.is_file():
            obj.log_error(f"Solvent {obj.solv_name} is not found")
            raise FileNotFoundError(f"Solvent {obj.solv_name} is not found")

        obj.template = parse_pdb_template(obj.data_path)
        obj.target_count = obj._target_solvent_count()
        obj._log_setup()
        obj._process()
        obj.log_info(["\n" + "-" * 70 + "\n"])
        return obj.atoms

    def _validate_options(self) -> None:
        if "write_cell" in self.params:
            raise ValueError(
                "Explicit solvent clusters are non-periodic; "
                "write_cell/PBC output is not supported."
            )
        if self.shape not in {"sphere", "cube"}:
            raise ValueError("Explicit solvent shape must be 'sphere' or 'cube'.")
        if self.shape == "sphere" and self.radius <= 0:
            raise ValueError("Explicit solvent radius must be > 0 for shape=sphere.")
        if self.shape == "cube":
            if self.box_size is None:
                raise ValueError("Explicit solvent shape=cube requires box_size.")
            self.box_size = float(self.box_size)
            if self.box_size <= 0:
                raise ValueError("Explicit solvent box_size must be > 0.")
        if self.density <= 0:
            raise ValueError("Explicit solvent density must be > 0.")
        if self.density_scale <= 0:
            raise ValueError("Explicit solvent density_scale must be > 0.")
        if self.number is not None and (type(self.number) is not int or self.number < 0):
            raise ValueError("Explicit solvent number must be an integer >= 0.")
        if self.clash_method not in {"vdw", "distance"}:
            raise ValueError("Explicit solvent clash_method must be 'vdw' or 'distance'.")
        if self.clash_method == "vdw":
            if "tolerance" in self.params or "clash_cutoff" in self.params:
                raise ValueError(
                    "Explicit solvent tolerance/clash_cutoff are only valid "
                    "with clash_method=distance."
                )
            if self.vdw_scale <= 0:
                raise ValueError("Explicit solvent vdw_scale must be > 0.")
            if self.vdw_fallback_radius <= 0:
                raise ValueError("Explicit solvent vdw_fallback_radius must be > 0.")
        elif self.tolerance <= 0:
            raise ValueError("Explicit solvent tolerance must be > 0.")
        if self.clash_method == "distance" and (
            "vdw_scale" in self.params or "vdw_fallback_radius" in self.params
        ):
            raise ValueError(
                "Explicit solvent vdw_scale/vdw_fallback_radius are only valid "
                "with clash_method=vdw."
            )
        if self.write_shell:
            if self.shell_cutoff is None:
                raise ValueError("Explicit solvent write_shell=true requires shell_cutoff.")
            self.shell_cutoff = float(self.shell_cutoff)
            if self.shell_cutoff <= 0:
                raise ValueError("Explicit solvent shell_cutoff must be > 0.")

    def _clash_method_label(self) -> str:
        if self.clash_method == "vdw":
            return (
                "vdw "
                f"(scale={self.vdw_scale:.4f}, "
                f"fallback_radius={self.vdw_fallback_radius:.3f} Å)"
            )
        return f"distance (tolerance={self.tolerance:.3f} Å)"

    def log_error(self, error_message: str) -> None:
        with open(self.output, "a", encoding="utf-8") as file:
            file.write(f"ERROR: {error_message}\n")

    def log_info(self, info_message: Iterable[str]) -> None:
        with open(self.output, "a", encoding="utf-8") as file:
            for info in info_message:
                file.write(str(info))

    def _volume(self) -> float:
        if self.shape == "sphere":
            return 4.0 / 3.0 * math.pi * self.radius ** 3
        return float(self.box_size) ** 3

    def _target_solvent_count(self) -> int:
        if self.number is not None:
            return int(self.number)
        first_group_symbols = [self.template.symbols[i] for i in self.template.groups[0]]
        if self.solv_name == "water":
            molar_mass = WATER_MOLAR_MASS_G_MOL
        else:
            molar_mass = _molecular_weight(first_group_symbols)
        number_density = _molecule_number_density(self.density, molar_mass) * self.density_scale
        return max(0, int(round(self._volume() * number_density)))

    def _extent(self) -> float:
        return self.radius if self.shape == "sphere" else float(self.box_size) / 2.0

    def _template_molecule_radius(self) -> float:
        return max(
            np.linalg.norm(
                self.template.coords[group] - self.template.coords[group].mean(axis=0),
                axis=1,
            ).max()
            for group in self.template.groups
        )

    def _molecule_centers(
        self,
        coords: np.ndarray,
        groups: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        return {tag: coords[indices].mean(axis=0) for tag, indices in groups.items()}

    @staticmethod
    def _groups_from_tags(tags: np.ndarray) -> dict[int, np.ndarray]:
        groups: dict[int, list[int]] = {}
        for idx, tag in enumerate(tags.tolist()):
            groups.setdefault(int(tag), []).append(idx)
        return {tag: np.asarray(indices, dtype=np.int64) for tag, indices in groups.items()}

    def _tile_template_network(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coords = self.template.coords
        symbols = self.template.symbols
        tags = self.template.residue_ids
        span = coords.max(axis=0) - coords.min(axis=0)
        span = np.where(span > 1.0e-6, span, 20.0)
        origin = coords.min(axis=0) + span / 2.0
        centered = coords - origin
        extent = self._extent()
        max_mol_radius = self._template_molecule_radius()
        tile_radius = max(0, int(math.ceil((extent + max_mol_radius) / float(span.min()))))
        if self.randomize:
            tile_radius = max(tile_radius, 1)
            phase = self.rng.random(3) * span
            rotation = _uniform_quaternion_rotation(self.rng)
        else:
            phase = np.zeros(3, dtype=np.float64)
            rotation = np.eye(3, dtype=np.float64)

        coords_all = []
        symbols_all: list[str] = []
        tags_all = []
        base_count = len(self.template.groups)
        tile_index = 0
        for i in range(-tile_radius, tile_radius + 1):
            for j in range(-tile_radius, tile_radius + 1):
                for k in range(-tile_radius, tile_radius + 1):
                    tile_shift = np.array([i, j, k], dtype=np.float64) * span
                    block = centered + tile_shift - phase
                    block = block @ rotation.T
                    coords_all.append(block)
                    symbols_all.extend(symbols.tolist())
                    tags_all.append(tags + base_count * tile_index)
                    tile_index += 1

        return (
            np.vstack(coords_all),
            np.asarray(symbols_all, dtype=object),
            np.concatenate(tags_all).astype(np.int64),
        )

    def _crop_tags(self, coords: np.ndarray, tags: np.ndarray) -> list[int]:
        groups = self._groups_from_tags(tags)
        centers = self._molecule_centers(coords, groups)
        keep: list[int] = []
        # Whole-molecule cropping uses a one-molecule-radius boundary allowance.
        # This avoids underfilling finite clusters simply because a retained
        # molecule must be kept/deleted as a whole residue.
        boundary_allowance = self._template_molecule_radius()
        if self.shape == "sphere":
            radius2 = (self.radius + boundary_allowance) ** 2
            for tag, center in centers.items():
                if float(center @ center) <= radius2:
                    keep.append(tag)
        else:
            half = float(self.box_size) / 2.0 + boundary_allowance
            for tag, center in centers.items():
                if bool((np.abs(center) <= half).all()):
                    keep.append(tag)
        return keep

    def _solute_clash_filter(
        self,
        coords: np.ndarray,
        symbols: np.ndarray,
        tags: np.ndarray,
        molecule_tags: list[int],
    ) -> list[int]:
        if not molecule_tags or len(self.atoms) == 0:
            return molecule_tags
        groups = self._groups_from_tags(tags)
        solute_positions = self.atoms.get_positions()
        solute_symbols = self.atoms.get_chemical_symbols()
        if self.clash_method == "vdw":
            solute_radii = _element_vdw_radii(solute_symbols, self.vdw_fallback_radius)
            solvent_radii = _element_vdw_radii(symbols.tolist(), self.vdw_fallback_radius)
        keep: list[int] = []
        for tag in molecule_tags:
            mol_indices = groups[tag]
            mol_coords = coords[mol_indices]
            if self.clash_method == "vdw":
                distances = np.linalg.norm(
                    mol_coords[:, None, :] - solute_positions[None, :, :],
                    axis=-1,
                )
                thresholds = (
                    solvent_radii[mol_indices, None] + solute_radii[None, :]
                ) * self.vdw_scale
                clashes = bool(np.any(distances < thresholds))
            else:
                distances = np.linalg.norm(
                    mol_coords[:, None, :] - solute_positions[None, :, :],
                    axis=-1,
                )
                clashes = float(distances.min()) < self.tolerance
            if not clashes:
                keep.append(tag)
        return keep

    def _cap_to_target(self, molecule_tags: list[int]) -> tuple[list[int], int]:
        if self.target_count <= 0:
            return [], len(molecule_tags)
        if len(molecule_tags) <= self.target_count:
            return molecule_tags, 0
        selected_positions = self.rng.choice(
            len(molecule_tags),
            size=self.target_count,
            replace=False,
        )
        selected = set(int(i) for i in selected_positions.tolist())
        capped = [tag for i, tag in enumerate(molecule_tags) if i in selected]
        return capped, len(molecule_tags) - len(capped)

    def _assemble_solvent(
        self, coords: np.ndarray, symbols: np.ndarray, tags: np.ndarray, molecule_tags: list[int]
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        if not molecule_tags:
            return np.empty((0, 3), dtype=np.float64), [], np.empty(0, dtype=np.int64)
        order = {tag: i for i, tag in enumerate(molecule_tags)}
        keep = np.isin(tags, molecule_tags)
        indices = np.where(keep)[0]
        indices = sorted(indices.tolist(), key=lambda idx: (order[int(tags[idx])], idx))
        final_indices = np.asarray(indices, dtype=np.int64)
        remap = {tag: i for i, tag in enumerate(molecule_tags)}
        final_tags = np.asarray([remap[int(tags[idx])] for idx in final_indices], dtype=np.int64)
        return coords[final_indices], symbols[final_indices].tolist(), final_tags

    def _set_nonperiodic_metadata(self, solvent_tags: np.ndarray) -> None:
        self.atoms.set_pbc([False, False, False])
        self.atoms.set_cell(np.zeros((3, 3)))
        molecule_ids = np.full(len(self.atoms), -1, dtype=np.int64)
        if len(solvent_tags) > 0:
            molecule_ids[self.solute_count:] = solvent_tags
        if "maple_molecule_id" in self.atoms.arrays:
            del self.atoms.arrays["maple_molecule_id"]
        self.atoms.new_array("maple_molecule_id", molecule_ids)
        self.atoms.info["maple_explicit_solvent"] = "non-periodic coordinate-only cluster"

    def _base_path(self) -> Path:
        return Path(os.path.splitext(self.output)[0])

    def _write_coordinate_outputs(self) -> tuple[Path, Path]:
        base = self._base_path()
        xyz_path = base.with_name(base.name + "_solvated.xyz")
        pdb_path = base.with_name(base.name + "_solvated.pdb")
        molecule_ids = self.atoms.arrays.get(
            "maple_molecule_id",
            np.full(len(self.atoms), -1, dtype=np.int64),
        )
        comment = "MAPLE explicit solvent cluster; non-periodic coordinate-only model"
        _write_xyz(xyz_path, self.atoms, comment)
        _write_pdb(pdb_path, self.atoms, self.solute_count, molecule_ids)
        return xyz_path, pdb_path

    def _shell_atom_indices(self) -> np.ndarray:
        if not self.write_shell:
            return np.arange(len(self.atoms), dtype=np.int64)
        molecule_ids = self.atoms.arrays["maple_molecule_id"]
        solute_indices = np.arange(self.solute_count, dtype=np.int64)
        solvent_indices = np.arange(self.solute_count, len(self.atoms), dtype=np.int64)
        if len(solvent_indices) == 0:
            return solute_indices
        solute_positions = self.atoms.positions[solute_indices]
        keep_molecules: set[int] = set()
        for molecule_id in sorted(set(molecule_ids[solvent_indices].tolist())):
            mol_indices = solvent_indices[molecule_ids[solvent_indices] == molecule_id]
            distances = np.linalg.norm(
                self.atoms.positions[mol_indices][:, None, :] - solute_positions[None, :, :],
                axis=-1,
            )
            if float(distances.min()) <= float(self.shell_cutoff):
                keep_molecules.add(int(molecule_id))
        keep_solvent = [idx for idx in solvent_indices if int(molecule_ids[idx]) in keep_molecules]
        return np.asarray(solute_indices.tolist() + keep_solvent, dtype=np.int64)

    def _write_shell_outputs(self) -> Optional[tuple[Path, Path, int]]:
        if not self.write_shell:
            return None
        indices = self._shell_atom_indices()
        cluster = self.atoms[indices]
        molecule_ids = self.atoms.arrays["maple_molecule_id"][indices]
        if "maple_molecule_id" in cluster.arrays:
            del cluster.arrays["maple_molecule_id"]
        cluster.new_array("maple_molecule_id", molecule_ids)
        cluster.set_pbc([False, False, False])
        cluster.set_cell(np.zeros((3, 3)))
        base = self._base_path()
        xyz_path = base.with_name(base.name + "_cluster.xyz")
        pdb_path = base.with_name(base.name + "_cluster.pdb")
        comment = (
            "MAPLE explicit solvent shell cluster; "
            f"cutoff={float(self.shell_cutoff):.3f} Å; non-periodic"
        )
        _write_xyz(xyz_path, cluster, comment)
        _write_pdb(pdb_path, cluster, self.solute_count, molecule_ids)
        solvent_molecules = len(set(mid for mid in molecule_ids.tolist() if mid >= 0))
        return xyz_path, pdb_path, solvent_molecules

    def _log_setup(self) -> None:
        if self.shape == "sphere":
            geometry = f"radius={self.radius:.3f} Å"
        else:
            geometry = f"box_size={float(self.box_size):.3f} Å"
        lines = [
            "\n\n" + "-" * 70 + "\n",
            f"{'Explicit Solvent Cluster Setup'.center(70)}\n\n",
            f"• Solvent type: {self.solv_name}\n",
            f"• Shape: {self.shape} ({geometry})\n",
            f"• Model: non-periodic coordinate-only cluster (no PBC/cell metadata)\n",
            f"• Density target: {self.density:.4f} g/mL × {self.density_scale:.4f}\n",
            f"• Target solvent molecules: {self.target_count}\n",
            f"• Solute-solvent clash method: {self._clash_method_label()}\n",
            f"• Randomize template sampling: {self.randomize}\n",
        ]
        if self.randomize:
            lines.append(
                "• Randomization: template phase shift + uniform-quaternion "
                "rigid rotation; solvent network geometry is not independently "
                "randomized\n"
            )
        if self.write_shell:
            lines.append(f"• Shell sidecar cutoff: {float(self.shell_cutoff):.3f} Å\n")
        lines.append("\n")
        self.log_info(lines)

    def _process(self):
        solute_center = self.atoms.get_positions().mean(axis=0)
        self.atoms.positions -= solute_center

        coords, symbols, tags = self._tile_template_network()
        candidate_count = len(set(tags.tolist()))
        cropped_tags = self._crop_tags(coords, tags)
        cropped_count = len(cropped_tags)
        clash_kept_tags = self._solute_clash_filter(coords, symbols, tags, cropped_tags)
        clash_removed = cropped_count - len(clash_kept_tags)
        final_tags, density_removed = self._cap_to_target(clash_kept_tags)

        solvent_positions, solvent_symbols, solvent_tags = self._assemble_solvent(
            coords,
            symbols,
            tags,
            final_tags,
        )
        if solvent_symbols:
            self.atoms += Atoms(symbols=solvent_symbols, positions=solvent_positions)
        self._set_nonperiodic_metadata(solvent_tags)

        xyz_path, pdb_path = self._write_coordinate_outputs()
        shell_result = self._write_shell_outputs()

        final_count = len(final_tags)
        actual_density = final_count / self._volume() if self._volume() > 0 else 0.0
        target_density = self.target_count / self._volume() if self._volume() > 0 else 0.0
        lines = [
            "Explicit solvent coordinate generation summary:\n",
            (
                f"candidate={candidate_count} cropped={cropped_count} "
                f"solute_clash_removed={clash_removed} "
                f"density_removed={density_removed} final={final_count}\n"
            ),
            f"target_number_density={target_density:.8f} molecules/Å^3\n",
            f"actual_number_density={actual_density:.8f} molecules/Å^3\n",
            f"solute_solvent_clash_method={self._clash_method_label()}\n",
            f"Added {len(solvent_positions)} solvent atoms.\n",
            f"Solvated XYZ written to: {xyz_path}\n",
            f"Solvated PDB written to: {pdb_path}\n",
        ]
        if final_count < self.target_count:
            lines.append(
                "WARNING: template sampling provided fewer non-clashing solvent "
                "molecules than the density target. Increase geometry size or "
                "lower the clash cutoff.\n"
            )
        if density_removed > 0:
            lines.append(
                "WARNING: solvent molecules were randomly subselected to match "
                "the density/number target; retained coordinates remain from "
                "the rigid solvent-network sample.\n"
            )
        if shell_result is not None:
            shell_xyz, shell_pdb, shell_molecules = shell_result
            lines.extend([
                f"Shell cluster solvent molecules: {shell_molecules}\n",
                f"Shell XYZ written to: {shell_xyz}\n",
                f"Shell PDB written to: {shell_pdb}\n",
            ])
        self.log_info(lines)
