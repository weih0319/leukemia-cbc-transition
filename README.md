# Short-term multianalyte CBC state transitions in leukemia

This repository contains the analysis and figure-generation code used for a three-centre retrospective study of observed 1-3-day complete blood count (CBC) transitions in leukemia. The primary framework classifies each complete test as C0 (no analytical threshold crossed), C1 (one crossed) or C2+ (two or more crossed).

The repository contains code only. Patient-level source data and derived row-level files are excluded. See [`data/README.md`](data/README.md) for the expected local inputs and privacy restrictions.

## Analysis definitions

The harmonised units are:

- WBC, PLT and absolute NEUT: x10^9/L
- HGB: g/L

The prespecified analytical thresholds are WBC >=100 x10^9/L, HGB <60 g/L, PLT <20 x10^9/L and NEUT <0.5 x10^9/L. These cutoffs define study states; they are not presented as verified centre-specific notification limits.

## Repository structure

- `prepare_master_tables.py`: source-column mapping, diagnosis handling and basic data harmonisation.
- `analysis/main_state_analysis.py`: source extraction, patient-day aggregation, adjacent-pair construction and primary state analyses.
- `analysis/patient_first_transition.py`: patient-level first-transition components and centre heterogeneity.
- `analysis/manuscript_completeness.py`: baseline, subgroup, missingness and sensitivity tables.
- `analysis/ml/`: patient-grouped model development and the exploratory external-centre assessment.
- `analysis/state_transitions/`: corrected estimands, linkage/subtype audits, bootstrap analyses and final figure scripts.
- `run_all.py`: executes the analysis in dependency order.

## Environment

The analysis was run with Python 3.11.2. Create an isolated environment and install the locked dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduction

After placing authorized source workbooks in a local directory, run:

```bash
python run_all.py --raw-dir data/raw
```

The default bootstrap counts and random seeds match the manuscript analysis. Runtime depends on hardware because the pipeline includes patient-cluster bootstrap and nested patient-grouped model selection.

The main execution order is:

1. Source mapping, daily aggregation and primary state-transition analysis.
2. Baseline, subgroup, missingness and threshold/time-window sensitivity analyses.
3. Patient-grouped exploratory model development and second-stage model comparison.
4. Corrected state estimands, linkage audits and stable-subtype analyses.
5. Main and supplementary figure generation.

Outputs are written under `analysis/formal_analysis_three_centers/`, `analysis/ml/`, `analysis/ml/round2/` and `analysis/state_transitions/`. These outputs are ignored by Git because some contain deidentified row-level data and must be reviewed before any release.

## Interpretation

The code describes observed complete retests among patients who were actually retested. It does not estimate untreated disease progression, a causal effect of retest timing or clinical utility. The two external-centre datasets had previously been inspected and therefore constitute exploratory transportability assessment rather than untouched independent validation.

## License

Code is released under the MIT License. The license does not apply to patient data, manuscript text, ethics documents or institutional materials.
