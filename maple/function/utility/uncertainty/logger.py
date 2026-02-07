from typing import List


def format_uncertainty_lines(
    eval_id: int,
    props: List[str],
    active_models: int,
    dropped_count: int,
    energy: dict = None,
    force: dict = None,
    observer_labels: List[str] = None,
    dropped_messages: List[str] = None,
) -> List[str]:
    lines: List[str] = []
    props_txt = ",".join(props)
    lines.append(
        f"[UNC] eval={eval_id} task=runtime active_models={active_models} "
        f"dropped={dropped_count} props={props_txt}\n"
    )

    if energy is not None:
        lines.append(
            "[UNC] E_primary={:.10f} Eh  E_mean={:.10f} Eh  E_std={:.3e} Eh  E_rms={:.3e} Eh\n".format(
                energy["primary"], energy["mean"], energy["std"], energy["rms"]
            )
        )
        if observer_labels is None:
            observer_labels = []
        for i, delta in enumerate(energy["deltas"], start=1):
            label = observer_labels[i - 1] if i - 1 < len(observer_labels) else f"observer{i}"
            lines.append(f"[UNC] dE({label}-primary)={delta:.3e} Eh\n")

    if force is not None:
        lines.append(
            "[UNC] F_sigma_rms={:.3e} Eh/Ang  F_sigma_max={:.3e} Eh/Ang\n".format(
                force["sigma_rms"], force["sigma_max"]
            )
        )

    if dropped_messages:
        for msg in dropped_messages:
            lines.append(f"[UNC][WARN] {msg}\n")

    return lines

