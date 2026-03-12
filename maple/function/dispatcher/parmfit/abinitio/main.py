'''
Author:  Kanbe 
Date: 2026-03-10 16:38:42
LastEditors:  Kanbe 
LastEditTime: 2026-03-10 20:12:16
FilePath: /MAPLE/maple/function/dispatcher/parmfit/abinitio/main.py
Description: 
'''
from utils import StructureRecognizer, ResidueRecognizer, CapBuilder, ModelCompound, FamilyTyper, TermEnumerator, DonorSeeder, MLPRefitter, Guesser

def parameterize_residue(pdb_path, target_key, family, cap_mode, parm_bundles, mlp_backend):
    # 1. recognize
    structure = StructureRecognizer().parse(pdb_path)
    target = ResidueRecognizer(structure).locate(target_key)
    context = ResidueRecognizer(structure).annotate_polymer_context(target)

    # 2. build model compound
    if cap_mode == "cap":
        model = CapBuilder().build_ace_nme_model(context)
    elif cap_mode == "neighbor":
        model = CapBuilder().build_ace_model(context)
    else:
        model = ModelCompound.from_residue(context.target)

    # 3. select primary database
    primary_db = parm_bundles[family]

    # 4. atom typing in ONE namespace
    typed_model, typed_residue = FamilyTyper(primary_db).assign(model, context)

    # 5. enumerate final terms on uncapped residue
    final_terms = TermEnumerator().enumerate_residue_terms(typed_residue, context)

    # 6. resolve parameters
    resolved = []
    for term in final_terms:
        p = primary_db.exact_match(term)
        if not p:
            p = primary_db.analog_match(term)
        if not p:
            p = DonorSeeder(parm_bundles).seed(term, primary_family=family)   # optional
        if not p and term.kind in {"bond", "angle"}:
            p = MLPRefitter(mlp_backend).fit_from_hessian(model, term)
        if not p and term.kind == "torsion" and term.is_rotatable:
            p = MLPRefitter(mlp_backend).fit_from_scan(model, term)
        if not p:
            p = Guesser().guess(term)
        resolved.append(p)

    # 7. charges / LJ
    charges = ChargeAssigner(primary_db).assign_or_fit(model, typed_residue, cap_mode)
    lj = LJResolver(primary_db, parm_bundles).resolve(typed_residue)

    # 8. export
    template = TemplateWriter().write(typed_residue, charges, context)
    patch = PatchWriter().write(resolved, lj)
    report = CoverageReporter().summarize(resolved)

    return template, patch, report