#!/usr/bin/env python3
"""Generate Gate 3 resource CSV and REVTeX table from committed HPs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_eeg.physical_resources import (  # noqa: E402
    buffer_resource_counts, conservative_measurement_counts, operation_counts,
)
from qrc_eeg.pipeline import kernel_for  # noqa: E402


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    cfg = json.loads((ROOT / "config/rotaA_gate3_frozen.json").read_text())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    selected = json.loads(subprocess.check_output(["git", "show", f"{commit}:results/eeg/hp_selected.json"], cwd=ROOT, text=True))
    hp = {"QRC_K0": {}, **{model: selected[model]["hp"] for model in cfg["resource_models"] if model != "QRC_K0"}}
    if canonical_hash(hp) != cfg["official_hp_sha256"]:
        raise SystemExit("INVALID_CONFIG: committed resource-model HP mapping changed")
    rows = []
    for model in cfg["resource_models"]:
        kernel = kernel_for(model, hp[model])
        base = buffer_resource_counts(4, kernel.K)
        ops = operation_counts(4, kernel.K, cfg["observables"], cfg["trajectory_length_for_resource_table"])
        measure = conservative_measurement_counts(cfg["observables"], 10000, cfg["trajectory_length_for_resource_table"])
        rows.append({
            "construction": model, **base, **ops,
            "observables": cfg["observables"],
            "measurement_groups_conservative": cfg["observables"],
            "measurement_groups_qwc": "not_implemented",
            "shots_per_group": "N_shots",
            "preparations_per_step_expression": f"{cfg['observables']}*N_shots",
            "preparations_per_step_at_10000": measure["preparations_per_step"],
            "trajectory_length": cfg["trajectory_length_for_resource_table"],
            "preparations_per_trajectory_at_10000": measure["preparations_per_trajectory"],
            "hp_json": json.dumps(hp[model], sort_keys=True),
            "git_commit": commit,
        })
    table = pd.DataFrame(rows)
    out = ROOT / "results/resources/qrc_resource_table.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    latex = [
        r"\begin{table*}[t]", r"\caption{Conservative resources of the simulated density-matrix reservoirs.}",
        r"\label{tab:physical_resources}", r"\begin{ruledtabular}",
        r"\begin{tabular}{lrrrrrr}",
        r"Construction & $K{+}1$ & Independent reals & Conservative reals & Buffer bytes & Mix proxy & $10^4$-shot preparations/step \\",
        r"\hline",
    ]
    for row in rows:
        latex.append(
            f"{row['construction'].replace('_', r'\_')} & {row['buffer_states']} & "
            f"{row['independent_real_parameters']} & {row['conservative_real_scalars']} & "
            f"{row['dense_buffer_bytes']} & {row['mix_complex_scalar_multiplies_per_step'] + row['mix_complex_additions_per_step']} & "
            f"{row['preparations_per_step_at_10000']} \\\\"
        )
    latex += [r"\end{tabular}", r"\end{ruledtabular}",
              r"\begin{flushleft}\footnotesize Buffer bytes use complex128. Measurement counts assume one independent group per Pauli observable; no gate decomposition or QWC optimization is claimed.\end{flushleft}",
              r"\end{table*}"]
    paper = ROOT / "paper/tab_physical_resources.tex"
    paper.write_text("\n".join(latex) + "\n")
    print(table[["construction", "buffer_states", "conservative_real_scalars", "dense_buffer_bytes"]].to_string(index=False))


if __name__ == "__main__":
    main()
