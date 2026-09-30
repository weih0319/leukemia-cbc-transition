from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


BASE = Path(__file__).resolve().parent / "formal_analysis_three_centers"
OUT = BASE / "manuscript_ready"
FIG = OUT / "figures"
PAIR_SOURCE = BASE / "deidentified_consecutive_pairs.csv.gz"
DAILY_SOURCE = BASE / "deidentified_daily_cbc.csv.gz"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

LABS = ["WBC", "HGB", "PLT", "NEUT"]
PRIMARY = {"WBC": 100.0, "HGB": 60.0, "PLT": 20.0, "NEUT": 0.5}
HIGH = {"WBC": True, "HGB": False, "PLT": False, "NEUT": False}
SCENARIOS = {
    "Primary": PRIMARY,
    "WBC 50": {**PRIMARY, "WBC": 50.0},
    "HGB 70": {**PRIMARY, "HGB": 70.0},
    "HGB 80": {**PRIMARY, "HGB": 80.0},
    "PLT 10": {**PRIMARY, "PLT": 10.0},
    "PLT 30": {**PRIMARY, "PLT": 30.0},
    "NEUT 1.0": {**PRIMARY, "NEUT": 1.0},
    "Broad": {"WBC": 50.0, "HGB": 80.0, "PLT": 30.0, "NEUT": 1.0},
}


def wilson(events: int, n: int) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    low, high = proportion_confint(events, n, method="wilson")
    return float(low), float(high)


def critical(values: pd.Series, lab: str, threshold: float) -> pd.Series:
    return values.ge(threshold) if HIGH[lab] else values.lt(threshold)


def first_cohort(pairs: pd.DataFrame, thresholds: dict[str, float], low_days: int = 1, high_days: int = 3) -> pd.DataFrame:
    complete = pairs[[*LABS, *[f"previous_{x}" for x in LABS]]].notna().all(axis=1)
    previous_any = pd.Series(False, index=pairs.index)
    current_any = pd.Series(False, index=pairs.index)
    for lab in LABS:
        previous_any |= critical(pairs[f"previous_{lab}"], lab, thresholds[lab])
        current_any |= critical(pairs[lab], lab, thresholds[lab])
    eligible = pairs.loc[
        complete & pairs["interval_days"].between(low_days, high_days) & ~previous_any
    ].copy()
    eligible["outcome"] = current_any.loc[eligible.index].astype(int)
    return (
        eligible.sort_values(["patient_id", "sample_date"])
        .drop_duplicates("patient_id", keep="first")
        .reset_index(drop=True)
    )


def fmt_median(series: pd.Series) -> str:
    q = series.quantile([0.25, 0.5, 0.75])
    return f"{q.loc[0.5]:.1f} ({q.loc[0.25]:.1f}–{q.loc[0.75]:.1f})"


def baseline_long(first: pd.DataFrame) -> pd.DataFrame:
    groups = [("Overall", first)] + [(center, group) for center, group in first.groupby("center", observed=True)]
    rows = []
    for label, group in groups:
        n = len(group)
        rows.extend(
            [
                {"center": label, "variable": "Patients", "summary": str(n)},
                {"center": label, "variable": "Age, years", "summary": fmt_median(group["age"])},
                {"center": label, "variable": "Sex code 2", "summary": f"{int(group['sex'].astype(str).eq('2').sum())} ({group['sex'].astype(str).eq('2').mean():.1%})"},
                {"center": label, "variable": "Baseline WBC", "summary": fmt_median(group["previous_WBC"])},
                {"center": label, "variable": "Baseline HGB", "summary": fmt_median(group["previous_HGB"])},
                {"center": label, "variable": "Baseline PLT", "summary": fmt_median(group["previous_PLT"])},
                {"center": label, "variable": "Baseline NEUT", "summary": fmt_median(group["previous_NEUT"])},
                {"center": label, "variable": "Repeat interval, days", "summary": fmt_median(group["interval_days"])},
                {"center": label, "variable": "Repeat at 1 day", "summary": f"{int(group['interval_days'].eq(1).sum())} ({group['interval_days'].eq(1).mean():.1%})"},
            ]
        )
    return pd.DataFrame(rows)


def baseline_by_outcome(first: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable, label in [
        ("age", "Age, years"),
        ("previous_WBC", "Baseline WBC"),
        ("previous_HGB", "Baseline HGB"),
        ("previous_PLT", "Baseline PLT"),
        ("previous_NEUT", "Baseline NEUT"),
        ("interval_days", "Repeat interval, days"),
    ]:
        no = first.loc[first["outcome"].eq(0), variable]
        yes = first.loc[first["outcome"].eq(1), variable]
        p = mannwhitneyu(no, yes, alternative="two-sided").pvalue
        rows.append({"variable": label, "no_transition": fmt_median(no), "transition": fmt_median(yes), "p_value": p, "test": "Mann-Whitney U"})
    for variable, label, positive in [("sex", "Sex code 2", "2"), ("interval_days", "Repeat at 1 day", 1)]:
        flag = first[variable].astype(str).eq(str(positive)) if variable == "sex" else first[variable].eq(positive)
        table = pd.crosstab(flag, first["outcome"]).reindex(index=[False, True], columns=[0, 1], fill_value=0)
        _, p = fisher_exact(table.to_numpy())
        summaries = []
        for outcome in [0, 1]:
            subset = flag.loc[first["outcome"].eq(outcome)]
            summaries.append(f"{int(subset.sum())} ({subset.mean():.1%})")
        rows.append({"variable": label, "no_transition": summaries[0], "transition": summaries[1], "p_value": p, "test": "Fisher exact"})
    return pd.DataFrame(rows)


def subgroup_tables(first: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = first.copy()
    work["age_group"] = pd.cut(work["age"], bins=[-np.inf, 17, 39, 59, np.inf], labels=["<18", "18–39", "40–59", "≥60"])
    work["sex_group"] = work["sex"].astype(str).map({"1": "Code 1", "2": "Code 2"}).fillna("Unknown")
    work["interval_group_primary"] = np.where(work["interval_days"].eq(1), "1 day", "2–3 days")
    definitions = [("center", "Centre"), ("age_group", "Age"), ("sex_group", "Sex"), ("interval_group_primary", "Repeat interval")]
    rows, tests = [], []
    for variable, label in definitions:
        table = pd.crosstab(work[variable], work["outcome"]).reindex(columns=[0, 1], fill_value=0)
        chi2, p, dof, expected = chi2_contingency(table)
        tests.append({"subgroup": label, "chi_square": chi2, "degrees_of_freedom": dof, "p_value": p, "minimum_expected_cell": float(expected.min()), "warning": "Small expected cell" if expected.min() < 5 else ""})
        for level, group in work.groupby(variable, observed=True):
            n, events = len(group), int(group["outcome"].sum())
            low, high = wilson(events, n)
            rows.append({"subgroup": label, "level": str(level), "patients": n, "events": events, "rate": events / n, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows), pd.DataFrame(tests)


def threshold_sensitivity(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, thresholds in SCENARIOS.items():
        cohort = first_cohort(pairs, thresholds)
        n, events = len(cohort), int(cohort["outcome"].sum())
        low, high = wilson(events, n)
        rows.append({"scenario": name, **thresholds, "patients": n, "events": events, "rate": events / n, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def interval_sensitivity(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, low_day, high_day in [("1 day", 1, 1), ("2–3 days", 2, 3), ("1–3 days primary", 1, 3), ("1–7 days", 1, 7)]:
        cohort = first_cohort(pairs, PRIMARY, low_day, high_day)
        n, events = len(cohort), int(cohort["outcome"].sum())
        low, high = wilson(events, n)
        rows.append({"window": label, "patients": n, "events": events, "rate": events / n, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def leave_one_center_out(first: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for omitted in first["center"].sort_values().unique():
        group = first.loc[first["center"].ne(omitted)]
        n, events = len(group), int(group["outcome"].sum())
        low, high = wilson(events, n)
        rows.append({"omitted_center": omitted, "patients": n, "events": events, "rate": events / n, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def pairwise_centers(first: pd.DataFrame) -> pd.DataFrame:
    centers = sorted(first["center"].unique())
    rows = []
    for i, first_center in enumerate(centers):
        for second_center in centers[i + 1 :]:
            group = first.loc[first["center"].isin([first_center, second_center])]
            counts = pd.crosstab(group["center"], group["outcome"]).reindex(index=[first_center, second_center], columns=[0, 1], fill_value=0)
            table = np.array(
                [
                    [counts.loc[first_center, 1], counts.loc[first_center, 0]],
                    [counts.loc[second_center, 1], counts.loc[second_center, 0]],
                ]
            )
            odds_ratio, p = fisher_exact(table)
            rows.append({"center_1": first_center, "center_2": second_center, "odds_ratio_center1_vs_center2": odds_ratio, "p_value_raw": p})
    adjusted = multipletests([row["p_value_raw"] for row in rows], method="holm")[1]
    for row, p_adj in zip(rows, adjusted):
        row["p_value_holm"] = p_adj
    return pd.DataFrame(rows)


def missingness(daily: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for center in sorted(daily["center"].unique()):
        d = daily.loc[daily["center"].eq(center)]
        p = pairs.loc[pairs["center"].eq(center)]
        row = {"center": center, "daily_records": len(d), "adjacent_pairs": len(p)}
        for lab in LABS:
            row[f"daily_{lab}_available"] = int(d[lab].notna().sum())
            row[f"daily_{lab}_available_rate"] = d[lab].notna().mean()
        pair_complete = p[[*LABS, *[f"previous_{x}" for x in LABS]]].notna().all(axis=1)
        row["complete_core_pairs"] = int(pair_complete.sum())
        row["complete_core_pair_rate"] = pair_complete.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def proximity_analysis(first: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = first.copy()
    work["near_WBC"] = work["previous_WBC"].between(50, 100, inclusive="left")
    work["near_HGB"] = work["previous_HGB"].between(60, 80, inclusive="left")
    work["near_PLT"] = work["previous_PLT"].between(20, 50, inclusive="left")
    work["near_NEUT"] = work["previous_NEUT"].between(0.5, 1.0, inclusive="left")
    work["near_count_raw"] = work[["near_WBC", "near_HGB", "near_PLT", "near_NEUT"]].sum(axis=1)
    work["near_count_group"] = work["near_count_raw"].clip(upper=2).map({0: "0", 1: "1", 2: "≥2"})
    rows = []
    for level, group in work.groupby("near_count_group", observed=True):
        n, events = len(group), int(group["outcome"].sum())
        low, high = wilson(events, n)
        rows.append({"near_critical_indicators": level, "patients": n, "events": events, "rate": events / n, "ci_low": low, "ci_high": high})
    model = smf.glm("outcome ~ near_count_raw + I(age / 10) + C(sex) + C(center) + C(interval_days)", data=work, family=sm.families.Binomial()).fit(cov_type="HC3")
    term = "near_count_raw"
    association = pd.DataFrame([{"term": "Per additional near-critical indicator", "odds_ratio": float(np.exp(model.params[term])), "ci_low": float(np.exp(model.conf_int().loc[term, 0])), "ci_high": float(np.exp(model.conf_int().loc[term, 1])), "p_value": float(model.pvalues[term]), "adjusted_for": "age, sex, centre, and 1–3-day interval"}])
    return pd.DataFrame(rows), association


def make_figures(flow: pd.DataFrame, components: pd.DataFrame, subgroups: pd.DataFrame, proximity: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.axis("off")
    y_positions = np.linspace(0.92, 0.08, len(flow))
    for i, (row, y) in enumerate(zip(flow.itertuples(index=False), y_positions)):
        ax.text(0.5, y, f"{row.stage}\n{int(row.count):,}", ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.45", facecolor="#e8f1f8", edgecolor="#2f5d7c"))
        if i < len(flow) - 1:
            ax.annotate("", xy=(0.5, y_positions[i + 1] + 0.045), xytext=(0.5, y - 0.045), arrowprops=dict(arrowstyle="->", color="#555555"))
    fig.tight_layout()
    fig.savefig(FIG / "figure_1_cohort_flow.png", dpi=300)
    plt.close(fig)

    plot = components.copy().sort_values("rate")
    fig, ax = plt.subplots(figsize=(7, 4.8))
    y = np.arange(len(plot))
    ax.errorbar(plot["rate"], y, xerr=[plot["rate"] - plot["ci_low"], plot["ci_high"] - plot["rate"]], fmt="o", capsize=4, color="#2f5d7c")
    ax.set_yticks(y, plot["label"])
    ax.set_xlabel("Patient-level first transition proportion")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(FIG / "figure_2_first_transition_components.png", dpi=300)
    plt.close(fig)

    centre = subgroups.loc[subgroups["subgroup"].eq("Centre")].sort_values("rate")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(centre))
    ax.errorbar(centre["rate"], y, xerr=[centre["rate"] - centre["ci_low"], centre["ci_high"] - centre["rate"]], fmt="o", capsize=4, color="#4c78a8")
    ax.set_yticks(y, centre["level"])
    ax.set_xlabel("First observed 1–3-day transition proportion")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(FIG / "figure_3_centre_estimates.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(proximity))
    ax.errorbar(x, proximity["rate"], yerr=[proximity["rate"] - proximity["ci_low"], proximity["ci_high"] - proximity["rate"]], fmt="o", capsize=5, color="#e45756")
    ax.set_xticks(x, proximity["near_critical_indicators"])
    ax.set_xlabel("Number of near-critical baseline indicators")
    ax.set_ylabel("First transition proportion")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(FIG / "figure_4_near_critical_gradient.png", dpi=300)
    plt.close(fig)


def main() -> None:
    pairs = pd.read_csv(PAIR_SOURCE, parse_dates=["sample_date"], low_memory=False)
    daily = pd.read_csv(DAILY_SOURCE, parse_dates=["sample_date"], low_memory=False)
    qc = pd.read_csv(BASE / "table_01_cohort_qc.csv")
    first = first_cohort(pairs, PRIMARY)
    if len(first) != 701 or int(first["outcome"].sum()) != 108:
        raise ValueError("Primary cohort QC failed")

    flow = pd.DataFrame(
        [
            {"stage": "Source records retained as definite leukemia", "count": int(qc["source_records"].sum())},
            {"stage": "Deidentified patients", "count": int(qc["patients"].sum())},
            {"stage": "Patient-date CBC records after same-day collapse", "count": int(qc["daily_records"].sum())},
            {"stage": "Adjacent-date CBC pairs", "count": int(qc["consecutive_pairs"].sum())},
            {"stage": "Pairs complete for WBC/HGB/PLT/NEUT", "count": int(qc["complete_core_pairs"].sum())},
            {"stage": "Patients with a first eligible 1–3-day pair", "count": len(first)},
            {"stage": "Patients with a new critical transition", "count": int(first["outcome"].sum())},
        ]
    )
    flow.to_csv(OUT / "table_01_cohort_flow.csv", index=False, encoding="utf-8-sig")
    baseline_long(first).to_csv(OUT / "table_02_baseline_by_centre.csv", index=False, encoding="utf-8-sig")
    baseline_by_outcome(first).to_csv(OUT / "table_03_baseline_by_outcome.csv", index=False, encoding="utf-8-sig")
    missingness(daily, pairs).to_csv(OUT / "table_04_missingness_by_centre.csv", index=False, encoding="utf-8-sig")
    subgroup, subgroup_tests = subgroup_tables(first)
    subgroup.to_csv(OUT / "table_05_patient_subgroups.csv", index=False, encoding="utf-8-sig")
    subgroup_tests.to_csv(OUT / "table_06_subgroup_heterogeneity.csv", index=False, encoding="utf-8-sig")
    threshold_sensitivity(pairs).to_csv(OUT / "table_07_patient_level_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    interval_sensitivity(pairs).to_csv(OUT / "table_08_patient_level_interval_sensitivity.csv", index=False, encoding="utf-8-sig")
    leave_one_center_out(first).to_csv(OUT / "table_09_leave_one_centre_out.csv", index=False, encoding="utf-8-sig")
    pairwise_centers(first).to_csv(OUT / "table_10_pairwise_centre_fisher.csv", index=False, encoding="utf-8-sig")
    proximity, proximity_model = proximity_analysis(first)
    proximity.to_csv(OUT / "table_11_near_critical_gradient.csv", index=False, encoding="utf-8-sig")
    proximity_model.to_csv(OUT / "table_12_near_critical_adjusted_association.csv", index=False, encoding="utf-8-sig")

    component_source = pd.read_csv(BASE / "table_10_patient_level_first_transition_components.csv")
    labels = {"any_critical": "Any critical", "single_critical": "Single critical", "multiple_critical": "Multiple critical", "new_WBC": "WBC ≥100", "new_HGB": "HGB <60", "new_PLT": "PLT <20", "new_NEUT": "NEUT <0.5"}
    components = component_source.loc[component_source["center"].eq("Overall")].copy()
    components["label"] = components["outcome"].map(labels)
    make_figures(flow, components, subgroup, proximity)

    primary = component_source.loc[(component_source["center"].eq("Overall")) & (component_source["outcome"].eq("any_critical"))].iloc[0]
    near_assoc = proximity_model.iloc[0]
    lines = [
        "# 正文完整性补充分析报告",
        "",
        "## 主终点复核",
        "",
        f"患者级首次1–3天危急转换为{int(primary.events)}/{int(primary.patients)}，{primary.rate:.1%}（95%CI {primary.ci_low:.1%}–{primary.ci_high:.1%}）。",
        "",
        "## 新增内容",
        "",
        "已补充研究对象流程、三中心基线特征、按结局基线比较、核心指标缺失率、年龄/性别/中心/间隔亚组、患者级阈值和时间窗敏感性、逐中心排除、中心两两Fisher精确检验及基线近危急指标数量分析。",
        "",
        "## 基线近危急指标梯度",
        "",
    ]
    for row in proximity.itertuples(index=False):
        lines.append(f"- {row.near_critical_indicators}项：{row.events}/{row.patients}，{row.rate:.1%}（95%CI {row.ci_low:.1%}–{row.ci_high:.1%}）。")
    lines.extend(
        [
            "",
            f"每增加一项近危急指标，与首次转换的调整后OR为{near_assoc.odds_ratio:.2f}（95%CI {near_assoc.ci_low:.2f}–{near_assoc.ci_high:.2f}，P={near_assoc.p_value:.3g}）；该结果为关联性次要分析，不是新的静态分层或临床评分。",
            "",
            "## 使用建议",
            "",
            "主文以患者级首次转换和三状态矩阵为核心。基线表、流程图、患者级阈值敏感性和逐中心排除应进入正文或主要补充材料。近危急指标梯度可作为有临床可解释性的次要结果。预测模型继续仅放补充材料。",
        ]
    )
    (OUT / "MANUSCRIPT_COMPLETENESS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
