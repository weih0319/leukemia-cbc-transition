from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    "analysis/main_state_analysis.py",
    "analysis/patient_first_transition.py",
    "analysis/manuscript_completeness.py",
    "analysis/ml/run_analysis.py",
    "analysis/ml/round2/run_round2.py",
    "analysis/state_transitions/run_state_reanalysis.py",
    "analysis/state_transitions/audit_cohort.py",
    "analysis/state_transitions/audit_linkage_sensitivity.py",
    "analysis/state_transitions/audit_tongling_semantics.py",
    "analysis/state_transitions/audit_matrix_sensitivity.py",
    "analysis/state_transitions/audit_subtype_labels.py",
    "analysis/state_transitions/analyze_subtype_transitions.py",
    "analysis/state_transitions/make_three_centre_flow.py",
    "analysis/state_transitions/make_four_group_figures.py",
    "analysis/state_transitions/make_supplementary_figures.py",
    "analysis/state_transitions/make_subtype_matrices_figure.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the leukemia CBC transition analysis pipeline.")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw", help="Directory containing the three authorized source workbooks.")
    parser.add_argument("--from-step", type=int, default=1, help="One-based first step to execute.")
    parser.add_argument("--to-step", type=int, default=len(SCRIPTS), help="One-based last step to execute.")
    args = parser.parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise SystemExit(f"Raw-data directory not found: {raw_dir}")
    expected = ["1.安医大.xlsx", "2.淮北煤炭.xlsx", "5.铜陵.xlsx"]
    missing = [name for name in expected if not (raw_dir / name).is_file()]
    if missing:
        raise SystemExit("Missing required source workbooks: " + ", ".join(missing))
    if not 1 <= args.from_step <= args.to_step <= len(SCRIPTS):
        raise SystemExit(f"Step range must satisfy 1 <= from-step <= to-step <= {len(SCRIPTS)}")
    env = os.environ.copy()
    env["LEUKEMIA_RAW_DIR"] = str(raw_dir)
    for number, relative in enumerate(SCRIPTS, start=1):
        if not args.from_step <= number <= args.to_step:
            continue
        script = ROOT / relative
        print(f"[{number}/{len(SCRIPTS)}] {relative}", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
