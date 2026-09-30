from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint


BASE = Path(__file__).resolve().parent / "formal_analysis_three_centers"
SOURCE = BASE / "deidentified_consecutive_pairs.csv.gz"
THRESHOLDS = {"WBC": 100.0, "HGB": 60.0, "PLT": 20.0, "NEUT": 0.5}
HIGH = {"WBC": True, "HGB": False, "PLT": False, "NEUT": False}


def is_critical(values: pd.Series, lab: str) -> pd.Series:
    return values.ge(THRESHOLDS[lab]) if HIGH[lab] else values.lt(THRESHOLDS[lab])


def wilson(events: int, n: int) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    low, high = proportion_confint(events, n, method="wilson")
    return float(low), float(high)


def main() -> None:
    data = pd.read_csv(SOURCE, low_memory=False)
    for lab in THRESHOLDS:
        previous_numeric = data[f"previous_{lab}"].copy()
        data[f"available_{lab}"] = data[lab].notna() & previous_numeric.notna()
        data[f"current_critical_{lab}"] = is_critical(data[lab], lab) & data[f"available_{lab}"]
        data[f"previous_critical_{lab}"] = is_critical(previous_numeric, lab) & data[f"available_{lab}"]
    data["complete"] = data[[f"available_{x}" for x in THRESHOLDS]].all(axis=1)
    data["previous_any"] = data[[f"previous_critical_{x}" for x in THRESHOLDS]].any(axis=1)
    eligible = data.loc[
        data["complete"] & data["interval_days"].between(1, 3) & ~data["previous_any"]
    ].copy()
    first = eligible.sort_values("sample_date").drop_duplicates("patient_id", keep="first")
    first["critical_count"] = first[
        [f"current_critical_{x}" for x in THRESHOLDS]
    ].sum(axis=1)

    rows = []
    for center in ["Overall", *sorted(first["center"].unique())]:
        group = first if center == "Overall" else first.loc[first["center"].eq(center)]
        n = len(group)
        outcomes = {
            "any_critical": group["critical_count"].gt(0),
            "single_critical": group["critical_count"].eq(1),
            "multiple_critical": group["critical_count"].ge(2),
            **{f"new_{lab}": group[f"current_critical_{lab}"] for lab in THRESHOLDS},
        }
        for outcome, values in outcomes.items():
            events = int(values.sum())
            low, high = wilson(events, n)
            rows.append(
                {
                    "center": center,
                    "outcome": outcome,
                    "patients": n,
                    "events": events,
                    "rate": events / n if n else np.nan,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    outcome_table = pd.DataFrame(rows)
    outcome_table.to_csv(
        BASE / "table_10_patient_level_first_transition_components.csv",
        index=False,
        encoding="utf-8-sig",
    )

    heterogeneity_rows = []
    for outcome in ["any_critical", "new_WBC", "new_HGB", "new_PLT", "new_NEUT"]:
        values = (
            first["critical_count"].gt(0)
            if outcome == "any_critical"
            else first[f"current_critical_{outcome.removeprefix('new_')}"]
        )
        table = pd.crosstab(first["center"], values).reindex(columns=[False, True], fill_value=0)
        chi2, p_value, dof, expected = chi2_contingency(table)
        min_expected = float(expected.min())
        heterogeneity_rows.append(
            {
                "outcome": outcome,
                "chi_square": chi2,
                "degrees_of_freedom": dof,
                "p_value": p_value,
                "minimum_expected_cell": min_expected,
                "interpretation_warning": (
                    "small expected cell; chi-square approximation unstable" if min_expected < 5 else ""
                ),
            }
        )
    pd.DataFrame(heterogeneity_rows).to_csv(
        BASE / "table_11_center_heterogeneity_patient_level.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall = outcome_table.loc[outcome_table["center"].eq("Overall")]
    labels = {
        "any_critical": "至少一种危急血象",
        "single_critical": "仅一种危急血象",
        "multiple_critical": "同时至少两种危急血象",
        "new_WBC": "新发WBC≥100",
        "new_HGB": "新发HGB<60",
        "new_PLT": "新发PLT<20",
        "new_NEUT": "新发NEUT<0.5",
    }
    lines = [
        "# 患者级首次危急转换补充结果（修正版）",
        "",
        f"纳入每名患者首次合格1–3天检测对，共{len(first):,}名患者。",
        "",
    ]
    for row in overall.itertuples(index=False):
        lines.append(
            f"- {labels[row.outcome]}：{row.events}/{row.patients}，{row.rate:.1%}（95% CI {row.ci_low:.1%}–{row.ci_high:.1%}）。"
        )
    lines.extend(
        [
            "",
            "单项事件可以同时发生，因此各单项人数之和可能大于任何危急转换人数。中心异质性检验仅为描述性，不能解释为中心医疗质量差异。",
            "",
            "QC：本修正版要求四项核心指标在前后两次检测均非缺失，纳入人数应与table_03_primary_endpoint.csv的Overall first_pair_patients完全一致。",
        ]
    )
    (BASE / "PATIENT_LEVEL_FIRST_TRANSITION_REPORT_REVISED.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
