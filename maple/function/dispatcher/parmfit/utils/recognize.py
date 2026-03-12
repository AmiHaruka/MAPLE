from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from typing import Optional

@dataclass(frozen=True)
class ResidueKey:
    model: int
    chain: str
    resseq: int
    icode: str
    resname: str

@dataclass
class Atom:
    serial: int
    record: str
    name: str
    altloc: str
    x: float
    y: float
    z: float
    occ: float
    bfac: float
    element: str = ""
    charge: str = ""
    residue_key: Optional[ResidueKey] = None

    def xyz(self):
        return (self.x, self.y, self.z)

@dataclass
class Residue:
    key: ResidueKey
    atoms: dict[str, Atom] = field(default_factory=dict)
    file_order: int = 0
    ter_after: bool = False
    parent_std: Optional[str] = None
    prev_parent_std: Optional[str] = None
    next_parent_std: Optional[str] = None
    ccd_type: Optional[str] = None

    def has(self, *names):
        return all(n in self.atoms for n in names)

    def __getitem__(self, name):
        return self.atoms[name]

@dataclass
class Structure:
    residues: list[Residue]
    links: list[tuple[ResidueKey, str, ResidueKey, str]] = field(default_factory=list)
    conect: dict[int, set[int]] = field(default_factory=lambda: defaultdict(set))
    serial_to_residue: dict[int, ResidueKey] = field(default_factory=dict)

def _safe_int(field: str, default: int = 0) -> int:
    text = field.strip()
    if not text:
        return default
    return int(text)


def _safe_float(field: str, default: float = 0.0) -> float:
    text = field.strip()
    if not text:
        return default
    return float(text)

def _normalize_pdb_line(line: str) -> str:
    return line.rstrip("\n").ljust(80)


def _parse_conect_line(line: str) -> tuple[Optional[int], list[int]]:
    serials: list[int] = []
    for i in range(6, len(line), 5):
        token = line[i:i + 5].strip()
        if not token:
            continue
        if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
            serials.append(int(token))
    if not serials:
        return None, []
    return serials[0], serials[1:]


def _parse_link_line(line: str, model: int) -> Optional[tuple[ResidueKey, str, ResidueKey, str]]:
    s = _normalize_pdb_line(line)
    try:
        rk1 = ResidueKey(
            model=model,
            chain=(s[21].strip() or "_"),
            resseq=_safe_int(s[22:26]),
            icode=s[26].strip(),
            resname=s[17:20].strip(),
        )
        rk2 = ResidueKey(
            model=model,
            chain=(s[51].strip() or "_"),
            resseq=_safe_int(s[52:56]),
            icode=s[56].strip(),
            resname=s[47:50].strip(),
        )
        a1 = s[12:16].strip()
        a2 = s[42:46].strip()
    except Exception:
        return None
    if not (a1 and a2 and rk1.resname and rk2.resname):
        return None
    return rk1, a1, rk2, a2

def parse_pdb_atom_line(line: str, record: str) -> Atom:
    s = _normalize_pdb_line(line)
    element = s[76:78].strip()
    atom_name = s[12:16].strip()
    return Atom(
        serial=_safe_int(s[6:11]),
        record=record,
        name=atom_name,
        altloc=s[16].strip(),
        x=_safe_float(s[30:38]),
        y=_safe_float(s[38:46]),
        z=_safe_float(s[46:54]),
        occ=_safe_float(s[54:60]),
        bfac=_safe_float(s[60:66]),
        element=(element),
        charge=s[78:80].strip(),
    )

def residue_key_from_line(line: str, model: int) -> ResidueKey:
    s = _normalize_pdb_line(line)
    return ResidueKey(
        model=model,
        chain=(s[21].strip() or "_"),
        resseq=_safe_int(s[22:26]),
        icode=s[26].strip(),
        resname=s[17:20].strip(),
    )

def distance(a: Atom, b: Atom) -> float:
    dx, dy, dz = a.x - b.x, a.y - b.y, a.z - b.z
    return (dx*dx + dy*dy + dz*dz) ** 0.5

def parse_pdb(path: str, keep_altloc: str = "A") -> Structure:
    residues_by_key: "OrderedDict[ResidueKey, Residue]" = OrderedDict()
    links: list[tuple[ResidueKey, str, ResidueKey, str]] = []
    conect: dict[int, set[int]] = defaultdict(set)
    serial_to_residue: dict[int, ResidueKey] = {}

    model = 1
    file_order = 0
    last_residue: Optional[Residue] = None

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = _normalize_pdb_line(raw)
            record = line[:6].strip().upper()

            if record == "MODEL":
                model = _safe_int(line[10:14], default=model)
                continue
            if record in {"ENDMDL", "END"}:
                continue
            if record == "TER":
                if last_residue is not None:
                    last_residue.ter_after = True
                last_residue = None
                continue
            if record == "LINK":
                link = _parse_link_line(line, model=model)
                if link:
                    links.append(link)
                continue
            if record == "CONECT":
                root, nbrs = _parse_conect_line(line)
                if root is not None:
                    for nbr in nbrs:
                        conect[root].add(nbr)
                        conect[nbr].add(root)
                continue
            if record not in {"ATOM", "HETATM", "HEATOM"}:
                continue

            atom = parse_pdb_atom_line(line, "HETATM" if record == "HEATOM" else record)
            if atom.altloc and atom.altloc not in {"A", keep_altloc}:
                continue

            rk = residue_key_from_line(line, model=model)
            atom.residue_key = rk

            residue = residues_by_key.get(rk)
            if residue is None:
                residue = Residue(key=rk, file_order=file_order)
                residues_by_key[rk] = residue
                file_order += 1

            existing = residue.atoms.get(atom.name)
            if existing is None or atom.occ >= existing.occ:
                residue.atoms[atom.name] = atom

            serial_to_residue[atom.serial] = rk
            last_residue = residue

    structure = Structure(
        residues=list(residues_by_key.values()),
        links=links,
        conect=conect,
        serial_to_residue=serial_to_residue,
    )
    annotate_parent_std(structure)
    return structure


def _has_conect_peptide(prev: Residue, curr: Residue, conect: dict[int, set[int]]) -> bool:
    if not prev.has("C") or not curr.has("N"):
        return False
    c_serial = prev["C"].serial
    n_serial = curr["N"].serial
    return n_serial in conect.get(c_serial, set())


def peptide_link(
    prev: Residue,
    curr: Residue,
    links,
    conect: Optional[dict[int, set[int]]] = None,
) -> bool:
    # explicit LINK first
    for rk1, a1, rk2, a2 in links:
        if rk1 == prev.key and a1 == "C" and rk2 == curr.key and a2 == "N":
            return True
        if rk2 == prev.key and a2 == "C" and rk1 == curr.key and a1 == "N":
            return True
    if conect and _has_conect_peptide(prev, curr, conect):
        return True
    if not prev.has("C") or not curr.has("N"):
        return False
    return 1.15 <= distance(prev["C"], curr["N"]) <= 1.50


def annotate_parent_std(structure: Structure) -> None:
    for i in range(len(structure.residues)):
        find_context(structure, i)


def _same_polymer_track(left: Residue, right: Residue) -> bool:
    return left.key.chain == right.key.chain and left.key.model == right.key.model


def find_context(structure: Structure, target_idx: int):
    n = len(structure.residues)
    target = structure.residues[target_idx]
    prev_res = structure.residues[target_idx - 1] if target_idx > 0 else None
    next_res = structure.residues[target_idx + 1] if target_idx + 1 < n else None

    if prev_res and (prev_res.ter_after or not _same_polymer_track(prev_res, target)):
        prev_res = None
    if next_res and (target.ter_after or not _same_polymer_track(next_res, target)):
        next_res = None

    if prev_res and not peptide_link(prev_res, target, structure.links, structure.conect):
        prev_res = None
    if next_res and not peptide_link(target, next_res, structure.links, structure.conect):
        next_res = None

    # For cyclic polymers: allow last<->first peptide closure when explicit TER is absent.
    if prev_res is None and target_idx == 0 and n > 1:
        wrap_prev = structure.residues[-1]
        if (
            not wrap_prev.ter_after
            and _same_polymer_track(wrap_prev, target)
            and peptide_link(wrap_prev, target, structure.links, structure.conect)
        ):
            prev_res = wrap_prev
    if next_res is None and target_idx == n - 1 and n > 1:
        wrap_next = structure.residues[0]
        if (
            not target.ter_after
            and _same_polymer_track(wrap_next, target)
            and peptide_link(target, wrap_next, structure.links, structure.conect)
        ):
            next_res = wrap_next

    target.prev_parent_std = prev_res.key.resname if prev_res else None
    target.next_parent_std = next_res.key.resname if next_res else None
    if target.prev_parent_std and target.next_parent_std:
        target.parent_std = f"{target.prev_parent_std}|{target.next_parent_std}"
    else:
        target.parent_std = target.prev_parent_std or target.next_parent_std

    return {
        "target": target,
        "prev_peptide": prev_res,
        "next_peptide": next_res,
        "n_open": prev_res is None,
        "c_open": next_res is None,
    }
