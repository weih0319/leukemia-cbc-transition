from __future__ import annotations

from pathlib import Path
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.stats.proportion import proportion_confint


ROOT = Path(__file__).resolve().parents[1]
BASE = Path(__file__).resolve().parent
OUT = BASE / "formal_analysis_three_centers"
FIG = OUT / "figures"
RAW = Path(os.environ.get("LEUKEMIA_RAW_DIR", ROOT / "data" / "raw")).resolve()
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

from prepare_master_tables import build_sheet_frame  # noqa: E402


SEED = 20260721
BOOTSTRAPS = 1000
CORE_LABS = ["WBC", "HGB", "PLT", "NEUT"]
FILES_AND_SHEETS = {
    "1.安医大.xlsx": ["Sheet1"],
    "2.淮北煤炭.xlsx": ["急性髓系", "急性淋系", "慢性髓系", "慢性淋系"],
    "5.铜陵.xlsx": ["AML", "ALL", "CML", "CLL"],
}
PRIMARY_THRESHOLDS = {"WBC": 100.0, "HGB": 60.0, "PLT": 20.0, "NEUT": 0.5}
HIGH_IS_CRITICAL = {"WBC": True, "HGB": False, "PLT": False, "NEUT": False}
OUTCOME_LABELS = {
    "WBC": "WBC >=100",
    "HGB": "HGB <60",
    "PLT": "PLT <20",
    "NEUT": "NEUT <0.5",
}
INTERVAL_ORDER = ["1 day", "2-3 days", "4-7 days", "8-14 days", "15-30 days", ">30 days"]


def clean_text(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out.isin(["", "nan", "None", "<NA>"]))


def parse_date(series: pd.Series) -> pd.Series:
    text = clean_text(series)
    digits = text.str.replace(r"\.0$", "", regex=True)
    ymd_mask = digits.str.fullmatch(r"\d{8}", na=False)
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    result.loc[ymd_mask] = pd.to_datetime(digits.loc[ymd_mask], format="%Y%m%d", errors="coerce")
    numeric = pd.to_numeric(text, errors="coerce")
    excel_mask = numeric.between(20_000, 80_000) & result.isna()
    result.loc[excel_mask] = pd.to_datetime(
        numeric.loc[excel_mask], unit="D", origin="1899-12-30", errors="coerce"
    )
    remaining = result.isna()
    result.loc[remaining] = pd.to_datetime(text.loc[remaining], errors="coerce")
    return result


def make_internal_key(frame: pd.DataFrame) -> pd.Series:
    name = clean_text(frame["name_raw"])
    patient_id = clean_text(frame["patient_id_raw"])
    sex = clean_text(frame["sex_raw"]).fillna("UNK")
    identity = name.map(lambda x: f"NAME:{x}" if pd.notna(x) else pd.NA)
    identity = identity.fillna(patient_id.map(lambda x: f"ID:{x}" if pd.notna(x) else pd.NA))
    return frame["center"].astype("string") + "|" + identity + "|SEX:" + sex


def critical(values: pd.Series, lab: str, threshold: float) -> pd.Series:
    if HIGH_IS_CRITICAL[lab]:
        return values.ge(threshold)
    return values.lt(threshold)


def load_source_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    extraction_rows = []
    for file_name, sheets in FILES_AND_SHEETS.items():
        path = RAW / file_name
        for sheet in sheets:
            frame, _ = build_sheet_frame(path, sheet)
            raw_n = len(frame)
            definite = frame["level2_type"].isin([0, 1, 2, 3])
            no_exclusion = frame["diagnosis_exclusion_reason"].astype(str).eq("")
            keep = definite & no_exclusion
            extraction_rows.append(
                {
                    "source_file": file_name,
                    "source_sheet": sheet,
                    "rows_read": raw_n,
                    "definite_leukemia_rows_retained": int(keep.sum()),
                    "rows_excluded": int((~keep).sum()),
                }
            )
            frames.append(frame.loc[keep].copy())
    data = pd.concat(frames, ignore_index=True)
    data["sample_date"] = parse_date(data["sample_date_raw"])
    data["internal_key"] = make_internal_key(data)
    data["age"] = pd.to_numeric(data["age"], errors="coerce")
    for lab in CORE_LABS:
        data[lab] = pd.to_numeric(data[lab], errors="coerce")

    validity = {
        "WBC": (0, 1000),
        "HGB": (1, 300),
        "PLT": (0, 5000),
        "NEUT": (0, 1000),
    }
    for lab, (lower, upper) in validity.items():
        bad = data[lab].notna() & ~data[lab].between(lower, upper)
        data[f"{lab}_implausible"] = bad
        data.loc[bad, lab] = np.nan

    valid_keys = sorted(data["internal_key"].dropna().unique())
    anon_map = {key: f"P{idx:06d}" for idx, key in enumerate(valid_keys, start=1)}
    data["patient_id"] = data["internal_key"].map(anon_map)
    return data, pd.DataFrame(extraction_rows)


def build_daily_and_pairs(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = data.dropna(subset=["patient_id", "sample_date"]).copy()
    same_day_counts = (
        eligible.groupby(["center", "patient_id", "sample_date"], observed=True)
        .size()
        .rename("source_rows_same_day")
        .reset_index()
    )
    daily_labs = (
        eligible.groupby(["center", "patient_id", "sample_date"], observed=True)[CORE_LABS]
        .median()
        .reset_index()
    )
    daily_meta = (
        eligible.groupby(["center", "patient_id", "sample_date"], observed=True)
        .agg(age=("age", "median"), sex=("sex", "first"))
        .reset_index()
    )
    daily = daily_labs.merge(daily_meta, on=["center", "patient_id", "sample_date"], how="left")
    daily = daily.merge(same_day_counts, on=["center", "patient_id", "sample_date"], how="left")
    daily = daily.sort_values(["center", "patient_id", "sample_date"]).reset_index(drop=True)
    grouped = daily.groupby(["center", "patient_id"], observed=True)
    daily["interval_days"] = grouped["sample_date"].diff().dt.days
    daily["prior_interval_days"] = grouped["interval_days"].shift(1)
    for lab in CORE_LABS:
        daily[f"previous_{lab}"] = grouped[lab].shift(1)
        daily[f"previous2_{lab}"] = grouped[lab].shift(2)
    pairs = daily.loc[daily["interval_days"].gt(0)].copy()
    pairs["interval_group"] = pd.cut(
        pairs["interval_days"],
        bins=[0, 1, 3, 7, 14, 30, np.inf],
        labels=INTERVAL_ORDER,
        include_lowest=True,
        ordered=True,
    )
    return eligible, daily, pairs


def add_threshold_states(pairs: pd.DataFrame, thresholds: dict[str, float], prefix: str = "") -> pd.DataFrame:
    out = pairs.copy()
    current_flags = []
    previous_flags = []
    for lab in CORE_LABS:
        available = out[lab].notna() & out[f"previous_{lab}"].notna()
        current_flag = critical(out[lab], lab, thresholds[lab]) & available
        previous_flag = critical(out[f"previous_{lab}"], lab, thresholds[lab]) & available
        out[f"{prefix}available_{lab}"] = available
        out[f"{prefix}current_critical_{lab}"] = current_flag
        out[f"{prefix}previous_critical_{lab}"] = previous_flag
        current_flags.append(f"{prefix}current_critical_{lab}")
        previous_flags.append(f"{prefix}previous_critical_{lab}")
    out[f"{prefix}complete_pair"] = out[[f"{prefix}available_{x}" for x in CORE_LABS]].all(axis=1)
    out[f"{prefix}current_critical_count"] = out[current_flags].sum(axis=1)
    out[f"{prefix}previous_critical_count"] = out[previous_flags].sum(axis=1)
    out[f"{prefix}current_any_critical"] = out[f"{prefix}current_critical_count"].gt(0)
    out[f"{prefix}previous_any_critical"] = out[f"{prefix}previous_critical_count"].gt(0)
    out[f"{prefix}new_any_critical"] = (
        ~out[f"{prefix}previous_any_critical"] & out[f"{prefix}current_any_critical"]
    )
    out[f"{prefix}previous_state"] = pd.cut(
        out[f"{prefix}previous_critical_count"], [-1, 0, 1, np.inf], labels=["C0", "C1", "C2+"]
    )
    out[f"{prefix}current_state"] = pd.cut(
        out[f"{prefix}current_critical_count"], [-1, 0, 1, np.inf], labels=["C0", "C1", "C2+"]
    )
    return out


def cluster_bootstrap_rates(frame: pd.DataFrame, event_col: str, n_boot: int = BOOTSTRAPS) -> dict[str, float]:
    if frame.empty:
        return {
            "pairs": 0,
            "events": 0,
            "patients": 0,
            "pair_rate": np.nan,
            "pair_ci_low": np.nan,
            "pair_ci_high": np.nan,
            "patient_equal_rate": np.nan,
            "patient_equal_ci_low": np.nan,
            "patient_equal_ci_high": np.nan,
        }
    patient = (
        frame.groupby("patient_id", observed=True)[event_col]
        .agg(events="sum", pairs="size")
        .reset_index(drop=True)
    )
    e = patient["events"].to_numpy(float)
    n = patient["pairs"].to_numpy(float)
    rng = np.random.default_rng(SEED + len(frame) + len(patient))
    idx = rng.integers(0, len(patient), size=(n_boot, len(patient)))
    pair_draws = e[idx].sum(axis=1) / n[idx].sum(axis=1)
    patient_rates = e / n
    patient_draws = patient_rates[idx].mean(axis=1)
    return {
        "pairs": int(n.sum()),
        "events": int(e.sum()),
        "patients": len(patient),
        "pair_rate": float(e.sum() / n.sum()),
        "pair_ci_low": float(np.quantile(pair_draws, 0.025)),
        "pair_ci_high": float(np.quantile(pair_draws, 0.975)),
        "patient_equal_rate": float(patient_rates.mean()),
        "patient_equal_ci_low": float(np.quantile(patient_draws, 0.025)),
        "patient_equal_ci_high": float(np.quantile(patient_draws, 0.975)),
    }


def primary_endpoint_table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for centre in ["Overall", *sorted(pairs["center"].unique())]:
        subset = pairs if centre == "Overall" else pairs.loc[pairs["center"].eq(centre)]
        eligible = subset.loc[
            subset["complete_pair"]
            & subset["interval_days"].between(1, 3)
            & ~subset["previous_any_critical"]
        ].copy()
        stats = cluster_bootstrap_rates(eligible, "new_any_critical")
        first = eligible.sort_values("sample_date").drop_duplicates("patient_id", keep="first")
        first_events = int(first["new_any_critical"].sum())
        first_n = len(first)
        ci = proportion_confint(first_events, first_n, alpha=0.05, method="wilson") if first_n else (np.nan, np.nan)
        rows.append(
            {
                "center": centre,
                **stats,
                "first_pair_patients": first_n,
                "first_pair_events": first_events,
                "first_pair_rate": first_events / first_n if first_n else np.nan,
                "first_pair_ci_low": ci[0],
                "first_pair_ci_high": ci[1],
            }
        )
    return pd.DataFrame(rows)


def individual_transition_table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lab in CORE_LABS:
        for interval in INTERVAL_ORDER:
            interval_data = pairs.loc[pairs["interval_group"].astype(str).eq(interval)]
            for centre in ["Overall", *sorted(pairs["center"].unique())]:
                subset = interval_data if centre == "Overall" else interval_data.loc[interval_data["center"].eq(centre)]
                subset = subset.loc[subset[f"available_{lab}"]].copy()
                new_base = subset.loc[~subset[f"previous_critical_{lab}"]].copy()
                new_base["event"] = new_base[f"current_critical_{lab}"]
                new_stats = cluster_bootstrap_rates(new_base, "event", n_boot=500)
                critical_base = subset.loc[subset[f"previous_critical_{lab}"]].copy()
                critical_base["persisted"] = critical_base[f"current_critical_{lab}"]
                persistence = cluster_bootstrap_rates(critical_base, "persisted", n_boot=500)
                rows.append(
                    {
                        "lab": lab,
                        "outcome": OUTCOME_LABELS[lab],
                        "interval_group": interval,
                        "center": centre,
                        "evaluable_pairs": len(subset),
                        "baseline_noncritical_pairs": new_stats["pairs"],
                        "new_critical_events": new_stats["events"],
                        "new_pair_rate": new_stats["pair_rate"],
                        "new_patient_equal_rate": new_stats["patient_equal_rate"],
                        "new_patient_equal_ci_low": new_stats["patient_equal_ci_low"],
                        "new_patient_equal_ci_high": new_stats["patient_equal_ci_high"],
                        "baseline_critical_pairs": persistence["pairs"],
                        "persistent_events": persistence["events"],
                        "persistence_pair_rate": persistence["pair_rate"],
                        "persistence_patient_equal_rate": persistence["patient_equal_rate"],
                        "persistence_patient_equal_ci_low": persistence["patient_equal_ci_low"],
                        "persistence_patient_equal_ci_high": persistence["patient_equal_ci_high"],
                    }
                )
    return pd.DataFrame(rows)


def state_transition_table(pairs: pd.DataFrame) -> pd.DataFrame:
    complete = pairs.loc[pairs["complete_pair"]].copy()
    complete["patient_interval_pairs"] = complete.groupby(
        ["patient_id", "interval_group"], observed=True
    )["patient_id"].transform("size")
    complete["patient_equal_weight"] = 1 / complete["patient_interval_pairs"]
    rows = []
    for interval in INTERVAL_ORDER:
        subset = complete.loc[complete["interval_group"].astype(str).eq(interval)]
        for previous in ["C0", "C1", "C2+"]:
            base = subset.loc[subset["previous_state"].astype(str).eq(previous)]
            denominator = len(base)
            weighted_denominator = base["patient_equal_weight"].sum()
            for current in ["C0", "C1", "C2+"]:
                cell = base.loc[base["current_state"].astype(str).eq(current)]
                rows.append(
                    {
                        "interval_group": interval,
                        "previous_state": previous,
                        "current_state": current,
                        "pairs": len(cell),
                        "pair_transition_probability": len(cell) / denominator if denominator else np.nan,
                        "patient_equal_transition_probability": (
                            cell["patient_equal_weight"].sum() / weighted_denominator
                            if weighted_denominator else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def threshold_sensitivity_table(base_pairs: pd.DataFrame) -> pd.DataFrame:
    scenarios = {
        "primary": {"WBC": 100, "HGB": 60, "PLT": 20, "NEUT": 0.5},
        "WBC_50": {"WBC": 50, "HGB": 60, "PLT": 20, "NEUT": 0.5},
        "HGB_70": {"WBC": 100, "HGB": 70, "PLT": 20, "NEUT": 0.5},
        "HGB_80": {"WBC": 100, "HGB": 80, "PLT": 20, "NEUT": 0.5},
        "PLT_10": {"WBC": 100, "HGB": 60, "PLT": 10, "NEUT": 0.5},
        "PLT_30": {"WBC": 100, "HGB": 60, "PLT": 30, "NEUT": 0.5},
        "NEUT_1": {"WBC": 100, "HGB": 60, "PLT": 20, "NEUT": 1.0},
        "broad": {"WBC": 50, "HGB": 80, "PLT": 30, "NEUT": 1.0},
    }
    rows = []
    for name, thresholds in scenarios.items():
        temp = add_threshold_states(base_pairs, thresholds, prefix="s_")
        eligible = temp.loc[
            temp["s_complete_pair"]
            & temp["interval_days"].between(1, 3)
            & ~temp["s_previous_any_critical"]
        ].copy()
        stats = cluster_bootstrap_rates(eligible, "s_new_any_critical", n_boot=500)
        rows.append({"scenario": name, **thresholds, **stats})
    return pd.DataFrame(rows)


def fit_dynamic_models(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    risk_rows = []
    centre_dummies_reference = "安医大"
    baseline_bins = {
        "WBC": [-np.inf, 10, 50, 100],
        "HGB": [60, 70, 80, 100, np.inf],
        "PLT": [20, 30, 50, 100, np.inf],
        "NEUT": [0.5, 1, 2, 5, np.inf],
    }
    labels = {
        "WBC": ["<10", "10-49", "50-99"],
        "HGB": ["60-69", "70-79", "80-99", ">=100"],
        "PLT": ["20-29", "30-49", "50-99", ">=100"],
        "NEUT": ["0.5-0.99", "1-1.99", "2-4.99", ">=5"],
    }
    for lab in CORE_LABS:
        frame = daily.loc[daily["interval_days"].between(1, 3)].copy()
        current = frame[lab]
        baseline = frame[f"previous_{lab}"]
        prior = frame[f"previous2_{lab}"]
        prior_interval = frame["prior_interval_days"]
        valid = (
            current.notna()
            & baseline.notna()
            & prior.notna()
            & prior_interval.between(1, 14)
        )
        frame = frame.loc[valid].copy()
        frame["event"] = critical(frame[lab], lab, PRIMARY_THRESHOLDS[lab]).astype(int)
        baseline_is_critical = critical(frame[f"previous_{lab}"], lab, PRIMARY_THRESHOLDS[lab])
        frame = frame.loc[~baseline_is_critical].copy()
        if HIGH_IS_CRITICAL[lab]:
            frame["distance"] = PRIMARY_THRESHOLDS[lab] - frame[f"previous_{lab}"]
            frame["trend_toward_critical"] = (
                frame[f"previous_{lab}"] - frame[f"previous2_{lab}"]
            ) / frame["prior_interval_days"]
        else:
            frame["distance"] = frame[f"previous_{lab}"] - PRIMARY_THRESHOLDS[lab]
            frame["trend_toward_critical"] = -(
                frame[f"previous_{lab}"] - frame[f"previous2_{lab}"]
            ) / frame["prior_interval_days"]

        for column in ["distance", "trend_toward_critical"]:
            lo, hi = frame[column].quantile([0.01, 0.99])
            frame[column] = frame[column].clip(lo, hi)
            sd = frame[column].std(ddof=0)
            frame[f"{column}_z"] = (frame[column] - frame[column].mean()) / sd if sd > 0 else 0.0
        frame["interval_z"] = frame["interval_days"] - frame["interval_days"].mean()
        centre = pd.get_dummies(frame["center"], prefix="center", drop_first=False, dtype=float)
        ref_col = f"center_{centre_dummies_reference}"
        if ref_col in centre:
            centre = centre.drop(columns=ref_col)
        x = pd.concat(
            [
                pd.Series(1.0, index=frame.index, name="intercept"),
                frame[["distance_z", "trend_toward_critical_z", "interval_z"]],
                centre,
            ],
            axis=1,
        ).astype(float)
        try:
            model = GEE(
                frame["event"].astype(float),
                x,
                groups=frame["patient_id"],
                family=Binomial(),
                cov_struct=Exchangeable(),
            ).fit(maxiter=100)
            for term in ["distance_z", "trend_toward_critical_z", "interval_z", *centre.columns]:
                coefficient = model.params.get(term, np.nan)
                se = model.bse.get(term, np.nan)
                rows.append(
                    {
                        "lab": lab,
                        "term": term,
                        "n_pairs": len(frame),
                        "n_patients": frame["patient_id"].nunique(),
                        "events": int(frame["event"].sum()),
                        "coefficient": coefficient,
                        "standard_error": se,
                        "odds_ratio": np.exp(coefficient),
                        "ci_low": np.exp(coefficient - 1.96 * se),
                        "ci_high": np.exp(coefficient + 1.96 * se),
                        "p_value": model.pvalues.get(term, np.nan),
                    }
                )
        except Exception as exc:
            warnings.warn(f"Dynamic GEE failed for {lab}: {exc}")

        frame["baseline_bin"] = pd.cut(
            frame[f"previous_{lab}"], bins=baseline_bins[lab], labels=labels[lab], right=False
        )
        try:
            frame["trend_group"] = pd.qcut(
                frame["trend_toward_critical"],
                q=3,
                labels=["slow_or_away", "intermediate", "rapid_toward"],
                duplicates="drop",
            )
        except ValueError:
            frame["trend_group"] = "all"
        for (base_bin, trend_group), group in frame.groupby(
            ["baseline_bin", "trend_group"], observed=True
        ):
            stats = cluster_bootstrap_rates(group, "event", n_boot=300)
            risk_rows.append(
                {
                    "lab": lab,
                    "baseline_bin": str(base_bin),
                    "trend_group": str(trend_group),
                    **stats,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(risk_rows)


def build_qc_tables(data: pd.DataFrame, eligible: pd.DataFrame, daily: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for centre, group in data.groupby("center", observed=True):
        e = eligible.loc[eligible["center"].eq(centre)]
        d = daily.loc[daily["center"].eq(centre)]
        p = pairs.loc[pairs["center"].eq(centre)]
        patients = d["patient_id"].nunique()
        patient_counts = d.groupby("patient_id", observed=True).size()
        rows.append(
            {
                "center": centre,
                "source_records": len(group),
                "missing_patient_key_records": int(group["patient_id"].isna().sum()),
                "dated_linkable_records": len(e),
                "patients": patients,
                "patients_ge2_dates": int((patient_counts >= 2).sum()),
                "daily_records": len(d),
                "same_day_extra_source_rows": int((d["source_rows_same_day"] - 1).clip(lower=0).sum()),
                "consecutive_pairs": len(p),
                "complete_core_pairs": int(p["complete_pair"].sum()) if "complete_pair" in p else np.nan,
                "first_date": d["sample_date"].min(),
                "last_date": d["sample_date"].max(),
                "median_dates_per_patient": patient_counts.median(),
                "p90_dates_per_patient": patient_counts.quantile(0.90),
            }
        )
    implausible = []
    for lab in CORE_LABS:
        for centre, group in data.groupby("center", observed=True):
            implausible.append(
                {"center": centre, "lab": lab, "implausible_values_set_missing": int(group[f"{lab}_implausible"].sum())}
            )
    return pd.DataFrame(rows), pd.DataFrame(implausible)


def make_figures(
    primary: pd.DataFrame,
    individual: pd.DataFrame,
    transitions: pd.DataFrame,
    risk_bins: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    forest = primary.loc[primary["center"].ne("Overall")].copy()
    forest = forest.sort_values("patient_equal_rate")
    y = np.arange(len(forest))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        forest["patient_equal_rate"],
        y,
        xerr=[
            forest["patient_equal_rate"] - forest["patient_equal_ci_low"],
            forest["patient_equal_ci_high"] - forest["patient_equal_rate"],
        ],
        fmt="o",
        capsize=4,
        color="#1f4e79",
    )
    overall = primary.loc[primary["center"].eq("Overall")].iloc[0]
    ax.axvline(overall["patient_equal_rate"], color="#c44e52", linestyle="--", label="Overall patient-equal")
    ax.set_yticks(y, forest["center"])
    ax.set_xlabel("Observed new critical-state transition probability within 3 days")
    ax.set_xlim(left=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "figure_1_center_forest.png", dpi=300)
    plt.close(fig)

    plot_data = individual.loc[individual["center"].eq("Overall")].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for lab in CORE_LABS:
        group = plot_data.loc[plot_data["lab"].eq(lab)].set_index("interval_group").reindex(INTERVAL_ORDER)
        ax.plot(INTERVAL_ORDER, group["new_patient_equal_rate"], marker="o", label=OUTCOME_LABELS[lab])
    ax.set_ylabel("Patient-equal observed new transition probability")
    ax.set_xlabel("Interval to next observed CBC")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "figure_2_new_critical_by_interval.png", dpi=300)
    plt.close(fig)

    short = transitions.loc[transitions["interval_group"].isin(["1 day", "2-3 days"])]
    matrix = (
        short.groupby(["previous_state", "current_state"], observed=True)
        .apply(
            lambda x: np.average(
                x["patient_equal_transition_probability"], weights=x["pairs"].clip(lower=1)
            ),
            include_groups=False,
        )
        .unstack()
        .reindex(index=["C0", "C1", "C2+"], columns=["C0", "C1", "C2+"])
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.1%}", ha="center", va="center", color="black")
    ax.set_xticks(range(3), matrix.columns)
    ax.set_yticks(range(3), matrix.index)
    ax.set_xlabel("Next observed state")
    ax.set_ylabel("Previous state")
    ax.set_title("Transitions within 3 days")
    fig.colorbar(im, ax=ax, label="Transition probability")
    fig.tight_layout()
    fig.savefig(FIG / "figure_3_three_state_transition_heatmap.png", dpi=300)
    plt.close(fig)

    labs = ["HGB", "PLT", "NEUT"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    for ax, lab in zip(axes, labs):
        group = risk_bins.loc[risk_bins["lab"].eq(lab)]
        matrix = group.pivot(index="baseline_bin", columns="trend_group", values="patient_equal_rate")
        desired_cols = [c for c in ["slow_or_away", "intermediate", "rapid_toward"] if c in matrix.columns]
        matrix = matrix.reindex(columns=desired_cols)
        im = ax.imshow(matrix, vmin=0, vmax=min(0.5, np.nanmax(matrix.to_numpy()) * 1.1), cmap="YlOrRd", aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.1%}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set_title(lab)
        ax.set_xlabel("Recent trend toward critical")
        ax.set_ylabel("Baseline value")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(FIG / "figure_4_baseline_and_slope_risk_heatmaps.png", dpi=300)
    plt.close(fig)


def format_rate(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.1%}"


def write_report(
    data: pd.DataFrame,
    daily: pd.DataFrame,
    pairs: pd.DataFrame,
    primary: pd.DataFrame,
    individual: pd.DataFrame,
    sensitivity: pd.DataFrame,
    dynamic_models: pd.DataFrame,
    qc: pd.DataFrame,
) -> None:
    overall = primary.loc[primary["center"].eq("Overall")].iloc[0]
    short_individual = individual.loc[
        individual["center"].eq("Overall")
        & individual["interval_group"].isin(["1 day", "2-3 days"])
    ].copy()
    short_summary = (
        short_individual.groupby(["lab", "outcome"], observed=True)
        .agg(
            baseline_noncritical_pairs=("baseline_noncritical_pairs", "sum"),
            new_events=("new_critical_events", "sum"),
        )
        .reset_index()
    )
    short_summary["pair_rate"] = short_summary["new_events"] / short_summary["baseline_noncritical_pairs"]
    sensitivity_min = sensitivity["patient_equal_rate"].min()
    sensitivity_max = sensitivity["patient_equal_rate"].max()

    lines = [
        "# 四中心白血病患者短期危急血象状态转换：正式分析报告",
        "",
        "生成日期：2026-07-21",
        "",
        "## 研究定位",
        "",
        "本研究不使用既往三级疾病分层。诊断信息仅用于确认白血病病例纳入；研究终点是同一患者相邻两次CBC之间的危急状态转换。",
        "",
        "## 数据流",
        "",
        f"- 四中心纳入原始记录：{len(data):,}。",
        f"- 去标识化患者：{daily['patient_id'].nunique():,}。",
        f"- 合并同日记录后患者-日期记录：{len(daily):,}。",
        f"- 相邻不同日期检测对：{len(pairs):,}。",
        f"- WBC/HGB/PLT/NEUT前后完整检测对：{int(pairs['complete_pair'].sum()):,}。",
        "",
        "## 主要终点",
        "",
        "主要终点为：基线无WBC≥100、HGB<60、PLT<20或NEUT<0.5，且在1–3天内实际接受下一次检测时，新出现至少一种危急血象。",
        "",
        f"- 合格检测对：{int(overall['pairs']):,}；事件：{int(overall['events']):,}。",
        f"- 检测对加权观察率：{format_rate(overall['pair_rate'])}（患者聚类bootstrap 95% CI {format_rate(overall['pair_ci_low'])}–{format_rate(overall['pair_ci_high'])}）。",
        f"- 患者等权观察率：{format_rate(overall['patient_equal_rate'])}（95% CI {format_rate(overall['patient_equal_ci_low'])}–{format_rate(overall['patient_equal_ci_high'])}）。",
        f"- 每名患者首次合格检测对：{int(overall['first_pair_events'])}/{int(overall['first_pair_patients'])}，{format_rate(overall['first_pair_rate'])}（Wilson 95% CI {format_rate(overall['first_pair_ci_low'])}–{format_rate(overall['first_pair_ci_high'])}）。",
        "",
        "患者等权结果是建议主报结果，因为各中心和各患者的检测频率差异明显。",
        "",
        "## 单项危急转换（1–3天合并的未调整检测对结果）",
        "",
    ]
    for row in short_summary.itertuples(index=False):
        lines.append(
            f"- {row.outcome}：{int(row.new_events):,}/{int(row.baseline_noncritical_pairs):,}，{row.pair_rate:.1%}。"
        )
    lines.extend(
        [
            "",
            "## 中心差异",
            "",
        ]
    )
    for row in primary.loc[primary["center"].ne("Overall")].itertuples(index=False):
        lines.append(
            f"- {row.center}：患者等权率 {format_rate(row.patient_equal_rate)}（95% CI {format_rate(row.patient_equal_ci_low)}–{format_rate(row.patient_equal_ci_high)}）；合格患者 {row.patients:,}。"
        )
    lines.extend(
        [
            "",
            "中心间差异既可能来自病例状态，也可能来自检测频率、治疗阶段和信息记录方式，不能解释为医疗质量差异。",
            "",
            "## 阈值敏感性",
            "",
            f"不同预设阈值情景下，患者等权观察率范围为 {format_rate(sensitivity_min)}–{format_rate(sensitivity_max)}。详细结果见 `table_06_threshold_sensitivity.csv`。",
            "",
            "## 基线距离与近期变化速度",
            "",
        ]
    )
    if dynamic_models.empty:
        lines.append("动态GEE模型未成功估计，需进一步检查数据结构。")
    else:
        key = dynamic_models.loc[dynamic_models["term"].isin(["distance_z", "trend_toward_critical_z"])]
        for row in key.itertuples(index=False):
            label = "距阈值每增加1 SD" if row.term == "distance_z" else "朝危急方向变化速度每增加1 SD"
            lines.append(
                f"- {row.lab}，{label}：OR {row.odds_ratio:.2f}（95% CI {row.ci_low:.2f}–{row.ci_high:.2f}，P={row.p_value:.3g}；n={row.n_pairs:,}检测对）。"
            )
    lines.extend(
        [
            "",
            "距离变量越大代表离危急阈值越远，理论上OR应小于1；趋势变量越大代表更快朝危急方向变化，理论上OR应大于1。以上为关联而非因果或临床预测验证。",
            "",
            "## 关键偏倚与限制",
            "",
            "1. 检测时点由病情决定，属于informative observation；结果仅适用于已经接受复查者。",
            "2. 缺少治疗、输血、感染、住院日和临床结局，不能把转换归因于治疗，也不能推导最佳复查间隔。",
            "3. 当前患者链接由中心内姓名或编号构建，正式投稿前必须由各中心核验一人多号和同名患者。输出数据已仅保留新生成的去标识化patient ID。",
            "4. 深圳的检测密度明显高于其他中心，因此必须同时报告检测对加权、患者等权和首次事件结果。",
            "5. 危急阈值是预先锁定的研究定义，不能等同于所有医院的正式危急值报告标准。",
            "",
            "## 可投稿核心结论",
            "",
            "在四中心真实世界白血病患者的已观察短期复查中，从非危急状态进入重度血细胞异常并不少见。患者等权分析、首次事件分析和多阈值敏感性结果用于判断该结论是否稳健；基线接近危急阈值及近期朝危急方向快速变化可作为下一步前瞻性动态监测研究的候选信号。",
            "",
            "## QC摘要",
            "",
        ]
    )
    for row in qc.itertuples(index=False):
        lines.append(
            f"- {row.center}：{row.patients:,}名患者，{row.daily_records:,}个患者-日期记录，{row.consecutive_pairs:,}个连续检测对；每名患者日期数中位数{row.median_dates_per_patient:.0f}。"
        )
    (OUT / "FORMAL_ANALYSIS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data, extraction = load_source_records()
    eligible, daily, base_pairs = build_daily_and_pairs(data)
    pairs = add_threshold_states(base_pairs, PRIMARY_THRESHOLDS)
    qc, implausible = build_qc_tables(data, eligible, daily, pairs)

    primary = primary_endpoint_table(pairs)
    individual = individual_transition_table(pairs)
    transitions = state_transition_table(pairs)
    sensitivity = threshold_sensitivity_table(base_pairs)
    dynamic_models, risk_bins = fit_dynamic_models(daily)

    extraction.to_csv(OUT / "table_00_source_extraction.csv", index=False, encoding="utf-8-sig")
    qc.to_csv(OUT / "table_01_cohort_qc.csv", index=False, encoding="utf-8-sig")
    implausible.to_csv(OUT / "table_02_implausible_values.csv", index=False, encoding="utf-8-sig")
    primary.to_csv(OUT / "table_03_primary_endpoint.csv", index=False, encoding="utf-8-sig")
    individual.to_csv(OUT / "table_04_individual_transitions.csv", index=False, encoding="utf-8-sig")
    transitions.to_csv(OUT / "table_05_state_transition_matrix.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(OUT / "table_06_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    dynamic_models.to_csv(OUT / "table_07_dynamic_gee_models.csv", index=False, encoding="utf-8-sig")
    risk_bins.to_csv(OUT / "table_08_baseline_slope_risk_bins.csv", index=False, encoding="utf-8-sig")

    # Deidentified analysis datasets; no names, raw identifiers, diagnoses, or source row keys are exported.
    daily_export = daily[
        ["center", "patient_id", "sample_date", "age", "sex", "source_rows_same_day", *CORE_LABS]
    ].copy()
    pair_export_columns = [
        "center",
        "patient_id",
        "sample_date",
        "age",
        "sex",
        "interval_days",
        "interval_group",
        *CORE_LABS,
        *[f"previous_{lab}" for lab in CORE_LABS],
        "complete_pair",
        "previous_critical_count",
        "current_critical_count",
        "previous_state",
        "current_state",
        "new_any_critical",
    ]
    daily_export.to_csv(OUT / "deidentified_daily_cbc.csv.gz", index=False, compression="gzip")
    pairs[pair_export_columns].to_csv(
        OUT / "deidentified_consecutive_pairs.csv.gz", index=False, compression="gzip"
    )

    make_figures(primary, individual, transitions, risk_bins)
    write_report(data, daily, pairs, primary, individual, sensitivity, dynamic_models, qc)

    manifest = pd.DataFrame(
        [
            {"item": "random_seed", "value": SEED},
            {"item": "cluster_bootstrap_replicates_primary", "value": BOOTSTRAPS},
            {"item": "primary_interval", "value": "1-3 days"},
            {"item": "primary_WBC_threshold", "value": PRIMARY_THRESHOLDS["WBC"]},
            {"item": "primary_HGB_threshold", "value": PRIMARY_THRESHOLDS["HGB"]},
            {"item": "primary_PLT_threshold", "value": PRIMARY_THRESHOLDS["PLT"]},
            {"item": "primary_NEUT_threshold", "value": PRIMARY_THRESHOLDS["NEUT"]},
        ]
    )
    manifest.to_csv(OUT / "analysis_manifest.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
