"""Small labelled-graph matcher for Amber residue templates."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .amber_templates import AmberResidueTemplate, AmberTemplateRegistry


@dataclass(frozen=True)
class ResidueTemplateMatch:
    template: AmberResidueTemplate
    atom_by_template_index: tuple[dict, ...]
    score: float


def _adjacency(nodes: Iterable[int], edges: set[tuple[int, int]]) -> dict[int, set[int]]:
    result = {node: set() for node in nodes}
    for left, right in edges:
        if left in result and right in result:
            result[left].add(right)
            result[right].add(left)
    return result


def _joint_wl_colors(
    template_elements: list[str],
    template_adjacency: dict[int, set[int]],
    input_elements: list[str],
    input_adjacency: dict[int, set[int]],
) -> tuple[dict[int, int], dict[int, int]]:
    graphs = ((template_elements, template_adjacency), (input_elements, input_adjacency))
    colors: list[dict[int, object]] = [
        {node: (elements[node], len(adjacency[node])) for node in adjacency}
        for elements, adjacency in graphs
    ]
    for _ in range(max(len(template_elements), 1)):
        signatures = [
            {
                node: (colors[index][node], tuple(sorted(colors[index][neighbor] for neighbor in adjacency[node])))
                for node in adjacency
            }
            for index, (_elements, adjacency) in enumerate(graphs)
        ]
        ordered = sorted({signature for graph in signatures for signature in graph.values()}, key=repr)
        ids = {signature: index for index, signature in enumerate(ordered)}
        refined = [{node: ids[signature] for node, signature in graph.items()} for graph in signatures]
        if refined == colors:
            break
        colors = refined
    return colors[0], colors[1]


def _mapping_score(template: AmberResidueTemplate, atoms: list[dict], mapping: dict[int, int]) -> float:
    score = 0.0
    for left, right in template.bonds:
        template_distance = np.linalg.norm(np.asarray(template.atoms[left].xyz) - np.asarray(template.atoms[right].xyz))
        input_distance = np.linalg.norm(np.asarray(atoms[mapping[left]]["xyz"]) - np.asarray(atoms[mapping[right]]["xyz"]))
        score += float((input_distance - template_distance) ** 2)
    return score


def _preserves_tetrahedral_chirality(
    template: AmberResidueTemplate,
    atoms: list[dict],
    mapping: dict[int, int],
    adjacency: dict[int, set[int]],
    colors: dict[int, int],
) -> bool:
    for center, neighbors in adjacency.items():
        if len(neighbors) != 4 or len({colors[neighbor] for neighbor in neighbors}) != 4:
            continue
        ordered = sorted(neighbors, key=lambda neighbor: (colors[neighbor], template.atoms[neighbor].name))
        template_center = np.asarray(template.atoms[center].xyz, dtype=float)
        input_center = np.asarray(atoms[mapping[center]]["xyz"], dtype=float)
        template_vectors = [np.asarray(template.atoms[neighbor].xyz, dtype=float) - template_center for neighbor in ordered[:3]]
        input_vectors = [np.asarray(atoms[mapping[neighbor]]["xyz"], dtype=float) - input_center for neighbor in ordered[:3]]
        template_volume = float(np.linalg.det(np.stack(template_vectors, axis=1)))
        input_volume = float(np.linalg.det(np.stack(input_vectors, axis=1)))
        if abs(template_volume) > 1.0e-4 and abs(input_volume) > 1.0e-4 and template_volume * input_volume < 0.0:
            return False
    return True


def _find_isomorphism(template: AmberResidueTemplate, atoms: list[dict], candidate_edges: set[tuple[int, int]]) -> ResidueTemplateMatch | None:
    template_edges = set(template.bonds)
    input_edges = {tuple(sorted(edge)) for edge in candidate_edges}
    if len(input_edges) < len(template_edges):
        return None
    template_adjacency = _adjacency(range(len(template.atoms)), template_edges)
    input_adjacency = _adjacency(range(len(atoms)), input_edges)
    template_elements = [atom.element for atom in template.atoms]
    input_elements = [atom["element"] for atom in atoms]
    template_colors, input_colors = _joint_wl_colors(
        template_elements,
        template_adjacency,
        input_elements,
        input_adjacency,
    )
    candidates = {
        node: [
            other
            for other in input_adjacency
            if template_elements[node] == input_elements[other]
            and len(template_adjacency[node]) <= len(input_adjacency[other])
            and not (
                Counter(template_elements[neighbor] for neighbor in template_adjacency[node])
                - Counter(input_elements[neighbor] for neighbor in input_adjacency[other])
            )
            and (
                len(input_edges) != len(template_edges)
                or template_colors[node] == input_colors[other]
            )
        ]
        for node in template_adjacency
    }
    if any(not values for values in candidates.values()):
        return None
    order = sorted(template_adjacency, key=lambda node: (len(candidates[node]), -len(template_adjacency[node]), node))
    mapping: dict[int, int] = {}
    used: set[int] = set()
    best: tuple[float, dict[int, int]] | None = None

    def visit(depth: int) -> bool:
        nonlocal best
        if depth == len(order):
            if not _preserves_tetrahedral_chirality(template, atoms, mapping, template_adjacency, template_colors):
                return False
            score = _mapping_score(template, atoms, mapping)
            best = (score, dict(mapping))
            return True
        node = order[depth]
        for other in candidates[node]:
            if other in used:
                continue
            if any(
                neighbor in template_adjacency[node] and mapped not in input_adjacency[other]
                for neighbor, mapped in mapping.items()
            ):
                continue
            mapping[node] = other
            used.add(other)
            found = visit(depth + 1)
            used.remove(other)
            del mapping[node]
            if found:
                return True
        return False

    visit(0)
    if best is None:
        return None
    score, best_mapping = best
    return ResidueTemplateMatch(
        template=template,
        atom_by_template_index=tuple(atoms[best_mapping[index]] for index in range(len(template.atoms))),
        score=score,
    )


def match_residue_template(
    residue: dict,
    candidate_pairs: set[tuple[int, int]],
    registry: AmberTemplateRegistry,
    *,
    disulfide_serials: set[int] | None = None,
) -> ResidueTemplateMatch | None:
    atoms = list(residue["atoms"])
    local_index = {atom["serial"]: index for index, atom in enumerate(atoms)}
    local_edges = {
        tuple(sorted((local_index[left], local_index[right])))
        for left, right in candidate_pairs
        if left in local_index and right in local_index
    }
    preferred = registry.templates_for_name(residue["resname"])
    candidates = preferred or registry.templates_for_elements([atom["element"] for atom in atoms])
    matches = [
        match
        for template in candidates
        if len(template.atoms) == len(atoms)
        if (match := _find_isomorphism(template, atoms, local_edges)) is not None
    ]
    source_resname = residue.get("source_resname", residue["resname"]).strip().upper()
    if not matches and preferred and source_resname not in {"CYM", "CYX"}:
        matches = [
            match
            for template in registry.templates_for_elements([atom["element"] for atom in atoms])
            if (match := _find_isomorphism(template, atoms, local_edges)) is not None
        ]
    if not matches:
        return None
    residue_serials = {atom["serial"] for atom in atoms}
    in_disulfide = bool(disulfide_serials) and not residue_serials.isdisjoint(disulfide_serials)

    def _selection_key(match: ResidueTemplateMatch) -> tuple:
        name = match.template.output_name.upper()
        # CYM (free thiolate) and CYX (disulfide) share an identical intra-residue
        # graph and near-identical reference geometry, so their distance scores tie
        # and cannot decide between them. Whether the SG is in a disulfide bond is
        # the real discriminator, so it must rank ahead of the geometric score.
        # The penalty is 0 for every other template, leaving normal scoring intact.
        if name in {"CYX", "CYM"}:
            disulfide_penalty = 0 if ((name == "CYX") == in_disulfide) else 1
        else:
            disulfide_penalty = 0
        return (disulfide_penalty, match.score, match.template.template_id)

    return min(matches, key=_selection_key)


def apply_template_match(residue: dict, match: ResidueTemplateMatch) -> set[tuple[int, int]]:
    template = match.template
    residue.setdefault("source_resname", residue["resname"])
    for index, atom in enumerate(match.atom_by_template_index):
        reference = template.atoms[index]
        atom.setdefault("source_name", atom["name"])
        atom["name"] = reference.name
        atom["amber_type"] = reference.amber_type
        atom["charge"] = reference.charge
        if reference.name in {"N", "CA", "C", "O", "OXT"}:
            atom["role"] = reference.name
    residue["atoms"] = list(match.atom_by_template_index)
    residue["coords"] = np.asarray([atom["xyz"] for atom in residue["atoms"]], dtype=float)
    residue["resname"] = template.output_name
    residue["template_id"] = template.template_id
    residue["template_category"] = template.category
    residue["net_charge"] = template.net_charge
    residue["kind"] = template.family
    residue["connect_atoms"] = tuple(
        match.atom_by_template_index[index]["serial"] if index >= 0 else None
        for index in template.connect
    )
    return {
        tuple(sorted((match.atom_by_template_index[left]["serial"], match.atom_by_template_index[right]["serial"])))
        for left, right in template.bonds
    }


def match_peptide_backbone(
    residue: dict,
    candidate_pairs: set[tuple[int, int]],
    *,
    previous_carbons: set[int] | None = None,
    next_nitrogens: set[int] | None = None,
) -> bool:
    atoms = list(residue["atoms"])
    by_serial = {atom["serial"]: atom for atom in atoms}
    adjacency = {serial: set() for serial in by_serial}
    for left, right in candidate_pairs:
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)

    choices: list[tuple[dict, dict, dict, list[dict]]] = []
    for carbonyl in atoms:
        if carbonyl["element"] != "C":
            continue
        oxygen_neighbors = [
            by_serial[serial]
            for serial in adjacency[carbonyl["serial"]]
            if by_serial[serial]["element"] == "O"
            and 1.05 <= float(np.linalg.norm(carbonyl["xyz"] - by_serial[serial]["xyz"])) <= 1.45
        ]
        if not oxygen_neighbors:
            continue
        for alpha_serial in adjacency[carbonyl["serial"]]:
            alpha = by_serial[alpha_serial]
            ca_c_distance = float(np.linalg.norm(carbonyl["xyz"] - alpha["xyz"]))
            if alpha["element"] != "C" or not 1.25 <= ca_c_distance <= 1.75:
                continue
            for nitrogen_serial in adjacency[alpha_serial]:
                nitrogen = by_serial[nitrogen_serial]
                n_ca_distance = float(np.linalg.norm(alpha["xyz"] - nitrogen["xyz"]))
                if nitrogen["element"] != "N" or not 1.20 <= n_ca_distance <= 1.70:
                    continue
                n_hydrogens = [
                    serial
                    for serial in adjacency[nitrogen_serial]
                    if by_serial[serial]["element"] == "H"
                ]
                if not n_hydrogens:
                    continue
                if previous_carbons and not any(
                    tuple(sorted((serial, nitrogen_serial))) in candidate_pairs
                    for serial in previous_carbons
                ):
                    continue
                if next_nitrogens and not any(
                    tuple(sorted((carbonyl["serial"], serial))) in candidate_pairs
                    for serial in next_nitrogens
                ):
                    continue
                choices.append((nitrogen, alpha, carbonyl, oxygen_neighbors))
    unique_choices = {
        (choice[0]["serial"], choice[1]["serial"], choice[2]["serial"]): choice
        for choice in choices
    }
    if len(unique_choices) != 1:
        return False

    nitrogen, alpha, carbonyl, oxygens = next(iter(unique_choices.values()))
    oxygens = sorted(oxygens, key=lambda atom: (float(np.linalg.norm(carbonyl["xyz"] - atom["xyz"])), atom["serial"]))
    assignments = {nitrogen["serial"]: "N", alpha["serial"]: "CA", carbonyl["serial"]: "C", oxygens[0]["serial"]: "O"}
    if len(oxygens) > 1 and float(np.linalg.norm(carbonyl["xyz"] - oxygens[1]["xyz"])) <= 1.45:
        assignments[oxygens[1]["serial"]] = "OXT"

    n_hydrogens = sorted(
        (by_serial[serial] for serial in adjacency[nitrogen["serial"]] if by_serial[serial]["element"] == "H"),
        key=lambda atom: atom["serial"],
    )
    for index, atom in enumerate(n_hydrogens, start=1):
        assignments[atom["serial"]] = "H" if len(n_hydrogens) == 1 else f"H{index}"
    ca_hydrogens = sorted(
        (by_serial[serial] for serial in adjacency[alpha["serial"]] if by_serial[serial]["element"] == "H"),
        key=lambda atom: atom["serial"],
    )
    for index, atom in enumerate(ca_hydrogens, start=1):
        assignments[atom["serial"]] = "HA" if len(ca_hydrogens) == 1 else f"HA{index}"

    used = set(assignments.values())
    for serial, role in assignments.items():
        atom = by_serial[serial]
        atom.setdefault("source_name", atom["name"])
        atom["name"] = role
        atom["role"] = role
    for atom in atoms:
        if atom["serial"] in assignments:
            continue
        atom.setdefault("source_name", atom["name"])
        name = atom["name"]
        if name.isalnum() and len(name) <= 4 and name not in used:
            used.add(name)
            continue
        base = atom["element"].upper()[:2]
        for index in range(1, 100):
            candidate = f"{base}{index}"[:4]
            if candidate not in used:
                atom["name"] = candidate
                used.add(candidate)
                break
    residue["connect_atoms"] = (nitrogen["serial"], carbonyl["serial"])
    return True
