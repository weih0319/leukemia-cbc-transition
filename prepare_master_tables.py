from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT
RAW_DIR = Path(os.environ.get("LEUKEMIA_RAW_DIR", ROOT / "data" / "raw")).resolve()
OUT_DIR = ROOT / "analysis" / "prepared_data"
OUT_DIR.mkdir(exist_ok=True)


TARGET_SHEETS = {"AML": 0, "ALL": 1, "CML": 2, "CLL": 3}
SHEET_ROLE_MAP = {
    "急性髓系": "AML",
    "急性淋系": "ALL",
    "慢性髓系": "CML",
    "慢性淋系": "CLL",
    "AML": "AML",
    "ALL": "ALL",
    "CML": "CML",
    "CLL": "CLL",
}


STANDARD_FEATURES = [
    "WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "RDWCV",
    "PLT", "MPV", "PDW", "PCT",
    "LYM%", "LYM", "MONO%", "MONO", "NEUT%", "NEUT", "BASO%", "BASO", "EO%", "EO",
    "PT", "INR", "APTT", "TT", "FIB", "DD",
    "LDH", "UA", "ALT", "AST", "ALP", "TBIL", "DBIL", "BUN", "TP", "ALB",
]


FEATURE_SELECTORS: dict[str, dict[str, list[str]]] = {
    "WBC": {"include": [r"^WBC$", r"白细胞.*WBC", r"白细胞计数"], "exclude": []},
    "RBC": {"include": [r"^RBC$", r"^\(HR\)红细胞计数$", r"红细胞\(RBC\)", r"^红细胞计数$"], "exclude": []},
    "HGB": {"include": [r"^HGB$", r"^\(HR\)血红蛋白$", r"血红蛋白\(HGB\)", r"^血红蛋白$"], "exclude": []},
    "HCT": {"include": [r"^HCT$", r"红细胞压积", r"红细胞比容"], "exclude": []},
    "MCV": {"include": [r"^MCV$", r"红细胞平均体积", r"平均红细胞体积"], "exclude": []},
    "MCH": {"include": [r"^MCH$", r"平均红细胞血红蛋白", r"平均RBC血红蛋白含量", r"平均血红蛋白量"], "exclude": []},
    "MCHC": {"include": [r"^MCHC$", r"平均血红蛋白浓度", r"平均RBC血红蛋白浓度"], "exclude": []},
    "RDWCV": {"include": [r"^RDW-?CV$", r"RDWCV", r"红细胞.*RDW-CV", r"红细胞分布宽度-?CV", r"红细胞分布宽度CV"], "exclude": [r"SD"]},
    "PLT": {"include": [r"^PLT$", r"血小板\(PLT\)", r"血小板计数"], "exclude": [r"手工"]},
    "MPV": {"include": [r"^MPV$", r"平均血小板体积"], "exclude": []},
    "PDW": {"include": [r"^PDW$", r"血小板分布宽度"], "exclude": []},
    "PCT": {"include": [r"^PCT$", r"PCT_XCG", r"血小板压积", r"血小板比积"], "exclude": []},
    "LYM%": {"include": [r"^LYMPH%$", r"^LY%$", r"淋巴细胞百分比", r"淋巴细胞比率", r"淋巴细胞%"], "exclude": [r"手工"]},
    "LYM": {"include": [r"^LYMPH#$", r"^LY#$", r"淋巴细胞绝对", r"淋巴细胞#"], "exclude": [r"手工"]},
    "MONO%": {"include": [r"^MONO%$", r"单核细胞百分比", r"单核细胞比率", r"单核细胞%"], "exclude": [r"手工"]},
    "MONO": {"include": [r"^MONO#$", r"单核细胞绝对", r"单核细胞#"], "exclude": [r"手工"]},
    "NEUT%": {"include": [r"^NEUT%$", r"^NE%$", r"中性粒细胞百分比", r"中性粒细胞比率", r"中性粒细胞%"], "exclude": [r"手工"]},
    "NEUT": {"include": [r"^NEUT#$", r"^NE#$", r"中性粒细胞绝对", r"中性粒细胞#"], "exclude": [r"手工"]},
    "BASO%": {"include": [r"^BASO%$", r"嗜碱性粒细胞百分比", r"嗜碱性粒细胞比率", r"嗜碱性粒细胞%"], "exclude": [r"手工"]},
    "BASO": {"include": [r"^BASO#$", r"嗜碱性粒细胞绝对", r"嗜碱性粒细胞#"], "exclude": [r"手工"]},
    "EO%": {"include": [r"^EO%$", r"嗜酸性粒细胞百分比", r"嗜酸性粒细胞比率", r"嗜酸性粒细胞%"], "exclude": [r"手工"]},
    "EO": {"include": [r"^EO#$", r"嗜酸性粒细胞绝对", r"嗜酸性粒细胞#"], "exclude": [r"手工"]},
    "PT": {"include": [r"^PT$", r"^凝血酶原时间$", r"^血浆凝血酶原时间$"], "exclude": [r"活动度", r"PT%", r"PT-R", r"参考", r"\.1"]},
    "INR": {"include": [r"^INR$", r"国际标准化"], "exclude": [r"\.1"]},
    "APTT": {"include": [r"^APTT$", r"^活化部分凝血活酶时间$"], "exclude": [r"比率", r"参考", r"\.1"]},
    "TT": {"include": [r"^TT$", r"^凝血酶时间$", r"^血浆凝血酶时间测定$"], "exclude": [r"比率", r"\.1"]},
    "FIB": {"include": [r"^FIB$", r"^FBG C$", r"^纤维蛋白原$", r"^纤维蛋白原含量$"], "exclude": [r"降解", r"FDP", r"\.1"]},
    "DD": {"include": [r"^D-D$", r"^DDPLUS$", r"^D-二聚体$", r"^D二聚体$"], "exclude": [r"急诊"]},
    "LDH": {"include": [r"^LDH$", r"乳酸脱氢酶"], "exclude": [r"胸腹水"]},
    "UA": {"include": [r"^UA$", r"^\(HR\)尿酸$", r"^尿酸", r"尿酸\(UA\)"], "exclude": []},
    "ALT": {"include": [r"^ALT$", r"丙氨酸氨基转移酶", r"谷丙转氨酶"], "exclude": []},
    "AST": {"include": [r"^AST$", r"门冬氨酸氨基转移酶", r"天门冬氨酸氨基转移酶", r"谷草转氨酶"], "exclude": [r"AST/ALT", r"线粒体"]},
    "ALP": {"include": [r"^ALP$", r"碱性磷酸酶"], "exclude": [r"胸腹水"]},
    "TBIL": {"include": [r"^TBIL$", r"总胆红素"], "exclude": []},
    "DBIL": {"include": [r"^DBIL$", r"直接胆红素"], "exclude": []},
    "BUN": {"include": [r"^BUN$", r"^尿素$", r"^尿素氮$"], "exclude": [r"\.1"]},
    "TP": {"include": [r"^TP$", r"^\(HR\)总蛋白$", r"^总蛋白$"], "exclude": [r"脑脊液", r"胸腹水"]},
    "ALB": {"include": [r"^ALB$", r"^\(HR\)白蛋白$", r"^白蛋白$", r"白蛋白\(ALB\)"], "exclude": [r"前白蛋白", r"胸腹水", r"尿微量", r"免疫比浊"]},
}


META_SELECTORS = {
    "patient_id_raw": ["PatiID", "病历号", "就诊号/住院号", "患者姓名", "姓名", "CName"],
    "name_raw": ["姓名", "患者姓名", "CName"],
    "sex_raw": ["性别", "CSex"],
    "age_raw": ["年龄", "CAge"],
    "diagnosis_raw": ["诊断", "Diagnose"],
    "sample_date_raw": ["审核日期", "检验日期", "报告日期", "ExamDate", "审核时间"],
    "sample_id_raw": ["样本号", "ExamSampleSYSID", "SampleNo"],
}


def clean_col(col: Any) -> str:
    return re.sub(r"\s+", " ", str(col).strip())


def center_from_file(path: Path) -> str:
    return re.sub(r"^\d+\.", "", path.stem)


def sheet_role(sheet: str) -> str:
    if sheet in SHEET_ROLE_MAP:
        return SHEET_ROLE_MAP[sheet]
    if sheet in {"Sheet1", "血液科数据-合并结果"} or sheet.startswith("结果查询"):
        return "mixed_or_summary"
    if "治疗" in sheet or "缓解" in sheet:
        return "treatment_or_remission"
    if "淋巴瘤" in sheet:
        return "lymphoma_other"
    if "急或慢性" in sheet:
        return "unclear_leukemia"
    return "other"


def find_first(cols: list[str], candidates: list[str]) -> str | None:
    for cand in candidates:
        if cand in cols:
            return cand
    return None


def find_feature_col(cols: list[str], feature: str) -> tuple[str | None, str]:
    selector = FEATURE_SELECTORS[feature]
    candidates = []
    for col in cols:
        if any(re.search(p, col, flags=re.IGNORECASE) for p in selector["include"]):
            if not any(re.search(p, col, flags=re.IGNORECASE) for p in selector["exclude"]):
                candidates.append(col)
    if not candidates:
        # Relax exclusion only when no clean candidate exists, still avoid clear non-target analytes.
        for col in cols:
            if any(re.search(p, col, flags=re.IGNORECASE) for p in selector["include"]):
                candidates.append(col)
    if not candidates:
        return None, "missing"
    return candidates[0], "unique" if len(candidates) == 1 else "multiple_used_first"


def override_feature_col(center: str, feature: str, cols: list[str], col: str | None, status: str) -> tuple[str | None, str]:
    """Center-specific column choices confirmed during manual QC."""
    if center == "\u6df1\u5733" and feature == "FIB":
        assay_col = "\u7ea4\u7ef4\u86cb\u767d\u539f\u6d4b\u5b9a"
        if assay_col in cols:
            return assay_col, "manual_qc_override_shenzhen_fib_assay"
    return col, status


def to_number(value: Any) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    text = text.replace(",", "")
    text = re.sub(r"[<>≤≥约]", "", text)
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_age(value: Any) -> float | None:
    return to_number(value)


def standardize_sex(value: Any) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"男", "m", "male", "1", "男 "}:
        return 1
    if text in {"女", "f", "female", "2", "女 "}:
        return 2
    return None


def infer_level2_from_diagnosis(text: Any) -> int | None:
    if pd.isna(text):
        return None
    s = str(text).upper().replace(" ", "")
    if "早幼粒" in s or "M3" in s or "APL" in s:
        return 0
    if "慢性淋巴" in s or "CLL" in s or "小淋巴细胞淋巴瘤" in s:
        return 3
    if ("慢性粒" in s or "慢性髓" in s or "BCR/ABL" in s or "BCR-ABL" in s or "PH1" in s or "CML" in s) and "粒单核" not in s:
        return 2
    if "急性淋巴" in s or "ALL" in s or "B-ALL" in s or "T-ALL" in s or "淋巴细胞白血病" in s:
        return 1
    if "急性髓" in s or "髓系白血病" in s or "髓样白血病" in s or "髓细胞白血病" in s or "髓性白血病" in s or "单核细胞白血病" in s or "AML" in s:
        return 0
    return None


def is_m3(text: Any) -> int:
    if pd.isna(text):
        return 0
    s = str(text).upper().replace(" ", "")
    return int("早幼粒" in s or "M3" in s or "APL" in s)


def exclusion_reason_from_diagnosis(text: Any) -> str:
    if pd.isna(text):
        return ""
    s = str(text).upper().replace(" ", "")
    patterns = [
        ("remission", r"缓解|完全缓解|伴缓解|CR\d|CR[，,)]"),
        ("post_treatment", r"治疗后|化疗后|移植后|术后"),
    ]
    reasons = [name for name, pat in patterns if re.search(pat, s)]
    return ";".join(reasons)


def standardize_hct(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    values = raw.map(to_number)
    std = values.copy()
    std = std.where(std.isna() | (std > 1.5), std * 100)
    flag = ((std.notna()) & ((std < 5) | (std > 80))).astype(int)
    flag = flag.mask(std.isna(), 0)
    return std, flag


def add_feature_flags(df: pd.DataFrame) -> pd.DataFrame:
    if "HCT_raw" in df.columns:
        hct_std, hct_flag = standardize_hct(df["HCT_raw"])
        df["HCT"] = hct_std
        df["HCT_unit_flag"] = hct_flag
    else:
        df["HCT_unit_flag"] = 0

    if "PCT" in df.columns:
        pct = pd.to_numeric(df["PCT"], errors="coerce")
        df["PCT_unit_flag"] = ((pct.notna()) & ((pct < 0) | (pct > 5))).astype(int)
    else:
        df["PCT_unit_flag"] = 0

    if "FIB" in df.columns:
        fib = pd.to_numeric(df["FIB"], errors="coerce")
        df["FIB_unit_flag"] = ((fib.notna()) & ((fib <= 0) | (fib > 20))).astype(int)
    else:
        df["FIB_unit_flag"] = 0

    percent_cols = ["LYM%", "MONO%", "NEUT%", "BASO%", "EO%"]
    if all(c in df.columns for c in percent_cols):
        pct_sum = sum(pd.to_numeric(df[c], errors="coerce") for c in percent_cols)
        df["diff_percent_sum"] = pct_sum
        df["diff_percent_flag"] = ((pct_sum.notna()) & ((pct_sum < 95) | (pct_sum > 105))).astype(int)
    else:
        df["diff_percent_sum"] = np.nan
        df["diff_percent_flag"] = 0
    return df


def build_sheet_frame(file_path: Path, sheet: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    center = center_from_file(file_path)
    role = sheet_role(sheet)
    raw = pd.read_excel(file_path, sheet_name=sheet)
    raw = raw.dropna(how="all").copy()
    raw.columns = [clean_col(c) for c in raw.columns]
    cols = list(raw.columns)

    out = pd.DataFrame(index=raw.index)
    out["center"] = center
    out["source_file"] = file_path.name
    out["source_sheet"] = sheet
    out["source_sheet_role"] = role

    for std_meta, candidates in META_SELECTORS.items():
        col = find_first(cols, candidates)
        out[std_meta] = raw[col] if col else np.nan

    out["age"] = out["age_raw"].map(parse_age)
    out["sex"] = out["sex_raw"].map(standardize_sex)

    mapping_rows = []
    for feature in STANDARD_FEATURES:
        col, status = find_feature_col(cols, feature)
        col, status = override_feature_col(center, feature, cols, col, status)
        mapping_rows.append(
            {
                "center": center,
                "source_file": file_path.name,
                "source_sheet": sheet,
                "source_sheet_role": role,
                "standard_feature": feature,
                "source_column_used": col or "",
                "mapping_status": status,
            }
        )
        if col:
            out[f"{feature}_raw"] = raw[col]
            if feature == "HCT":
                # Standardized later while preserving raw.
                out[feature] = raw[col].map(to_number)
            else:
                out[feature] = raw[col].map(to_number)
        else:
            out[f"{feature}_raw"] = np.nan
            out[feature] = np.nan

    if role in TARGET_SHEETS:
        out["level2_type"] = TARGET_SHEETS[role]
        out["label_source"] = "sheet"
    elif center == "安医大" and "group" in raw.columns:
        # 安医大 group: 0=AML, 1=ALL, 2=CLL, 3=CML, 4=unclear acute leukemia.
        group = raw["group"]
        out["level2_type"] = group.map({0: 0, 1: 1, 3: 2, 2: 3})
        out["label_source"] = "anyida_group"
    else:
        out["level2_type"] = out["diagnosis_raw"].map(infer_level2_from_diagnosis)
        out["label_source"] = "diagnosis_keyword"

    out["level1_status"] = np.where(out["level2_type"].notna(), 1, np.nan)
    out["level3_aml_m3"] = np.where(out["level2_type"].eq(0), out["diagnosis_raw"].map(is_m3), np.nan)
    out["diagnosis_exclusion_reason"] = out["diagnosis_raw"].map(exclusion_reason_from_diagnosis)
    out = add_feature_flags(out)

    return out, pd.DataFrame(mapping_rows)


def main() -> None:
    frames = []
    mappings = []
    excluded = []

    for file_path in sorted(RAW_DIR.glob("*.xlsx")):
        xls = pd.ExcelFile(file_path)
        for sheet in xls.sheet_names:
            role = sheet_role(sheet)
            if role in {"treatment_or_remission", "lymphoma_other", "unclear_leukemia"}:
                continue
            # Avoid duplicating center-specific mixed summaries when disease-specific sheets exist.
            if role == "mixed_or_summary" and center_from_file(file_path) != "安医大":
                continue
            frame, mapping = build_sheet_frame(file_path, sheet)
            mappings.append(mapping)

            target_mask = frame["level2_type"].isin([0, 1, 2, 3])
            exclusion_mask = frame["diagnosis_exclusion_reason"].astype(str).ne("")
            keep = target_mask & ~exclusion_mask
            frame["included_in_master"] = keep.astype(int)
            frame["row_exclusion_reason"] = ""
            frame.loc[~target_mask, "row_exclusion_reason"] = "not_target_label"
            frame.loc[target_mask & exclusion_mask, "row_exclusion_reason"] = frame.loc[target_mask & exclusion_mask, "diagnosis_exclusion_reason"]

            frames.append(frame.loc[keep].copy())
            excluded.append(frame.loc[~keep].copy())

    master = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    excluded_df = pd.concat(excluded, ignore_index=True) if excluded else pd.DataFrame()
    mapping_df = pd.concat(mappings, ignore_index=True) if mappings else pd.DataFrame()

    master["level2_type"] = master["level2_type"].astype("Int64")
    master["level1_status"] = master["level1_status"].astype("Int64")
    master["level3_aml_m3"] = master["level3_aml_m3"].astype("Int64")

    id_cols = ["center", "source_file", "source_sheet", "source_sheet_role", "patient_id_raw", "sample_id_raw", "name_raw", "sex_raw", "sex", "age_raw", "age", "sample_date_raw", "diagnosis_raw", "label_source", "level1_status", "level2_type", "level3_aml_m3"]
    flag_cols = ["HCT_unit_flag", "PCT_unit_flag", "FIB_unit_flag", "diff_percent_sum", "diff_percent_flag"]
    feature_cols = []
    for feature in STANDARD_FEATURES:
        feature_cols.extend([f"{feature}_raw", feature])
    ordered = [c for c in id_cols + feature_cols + flag_cols if c in master.columns]
    master = master[ordered]

    master.to_csv(OUT_DIR / "master_target_labeled.csv", index=False, encoding="utf-8-sig")
    excluded_df.to_csv(OUT_DIR / "excluded_rows_audit.csv", index=False, encoding="utf-8-sig")
    mapping_df.to_csv(OUT_DIR / "standard_feature_mapping_used.csv", index=False, encoding="utf-8-sig")

    summary = build_summary(master, excluded_df, mapping_df)
    summary.to_csv(OUT_DIR / "master_summary.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUT_DIR / "prepared_data_summary.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        mapping_df.to_excel(writer, sheet_name="mapping_used", index=False)
        master.head(1000).to_excel(writer, sheet_name="master_head1000", index=False)
        excluded_df[["center", "source_file", "source_sheet", "diagnosis_raw", "row_exclusion_reason"]].head(1000).to_excel(writer, sheet_name="excluded_head1000", index=False)

    write_report(master, excluded_df, mapping_df, summary)


def build_summary(master: pd.DataFrame, excluded_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (center, sheet, role), g in master.groupby(["center", "source_sheet", "source_sheet_role"], dropna=False):
        counts = g["level2_type"].value_counts(dropna=False).sort_index()
        rows.append(
            {
                "center": center,
                "source_sheet": sheet,
                "source_sheet_role": role,
                "included_n": len(g),
                "AML_0": int(counts.get(0, 0)),
                "ALL_1": int(counts.get(1, 0)),
                "CML_2": int(counts.get(2, 0)),
                "CLL_3": int(counts.get(3, 0)),
                "AML_M3_1": int(g["level3_aml_m3"].eq(1).sum()),
                "AML_non_M3_0": int(g["level3_aml_m3"].eq(0).sum()),
                "HCT_flags": int(g["HCT_unit_flag"].sum()) if "HCT_unit_flag" in g else 0,
                "PCT_flags": int(g["PCT_unit_flag"].sum()) if "PCT_unit_flag" in g else 0,
                "FIB_flags": int(g["FIB_unit_flag"].sum()) if "FIB_unit_flag" in g else 0,
                "diff_percent_flags": int(g["diff_percent_flag"].sum()) if "diff_percent_flag" in g else 0,
            }
        )
    excluded_reason = excluded_df["row_exclusion_reason"].value_counts(dropna=False).to_dict() if not excluded_df.empty else {}
    rows.append(
        {
            "center": "__TOTAL__",
            "source_sheet": "__ALL__",
            "source_sheet_role": "__ALL__",
            "included_n": len(master),
            "AML_0": int(master["level2_type"].eq(0).sum()),
            "ALL_1": int(master["level2_type"].eq(1).sum()),
            "CML_2": int(master["level2_type"].eq(2).sum()),
            "CLL_3": int(master["level2_type"].eq(3).sum()),
            "AML_M3_1": int(master["level3_aml_m3"].eq(1).sum()),
            "AML_non_M3_0": int(master["level3_aml_m3"].eq(0).sum()),
            "HCT_flags": int(master["HCT_unit_flag"].sum()) if "HCT_unit_flag" in master else 0,
            "PCT_flags": int(master["PCT_unit_flag"].sum()) if "PCT_unit_flag" in master else 0,
            "FIB_flags": int(master["FIB_unit_flag"].sum()) if "FIB_unit_flag" in master else 0,
            "diff_percent_flags": int(master["diff_percent_flag"].sum()) if "diff_percent_flag" in master else 0,
            "excluded_reason_counts": str(excluded_reason),
        }
    )
    return pd.DataFrame(rows)


def write_report(master: pd.DataFrame, excluded_df: pd.DataFrame, mapping_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = []
    lines.append("# 标准化主分析数据整理报告")
    lines.append("")
    lines.append("Data source: authorized local workbooks supplied through LEUKEMIA_RAW_DIR.")
    lines.append("")
    lines.append("## 1. 纳入与排除规则")
    lines.append("")
    lines.append("纳入：")
    lines.append("")
    lines.append("- 安医大 `Sheet1` 中 `group=0/1/2/3` 可明确映射为 AML/ALL/CLL/CML 的记录。")
    lines.append("- 淮北煤炭、深圳、徐汇中心、铜陵的 AML/ALL/CML/CLL 疾病分 sheet。")
    lines.append("")
    lines.append("排除：")
    lines.append("")
    lines.append("- `治疗`、`缓解` sheet。")
    lines.append("- `淋巴瘤`、`急或慢性白血病` 等非目标 sheet。")
    lines.append("- 其他中心的混合总表，避免与疾病分 sheet 重复。")
    lines.append("- 诊断文本中提示缓解/治疗后/化疗后/移植后/CR 状态的记录。")
    lines.append("- 不能明确映射为 AML/ALL/CML/CLL 的记录。")
    lines.append("")
    lines.append("## 2. 主表样本量")
    lines.append("")
    total = summary[summary["center"].eq("__TOTAL__")].iloc[0]
    lines.append(f"- 总纳入记录数：{int(total['included_n'])}")
    lines.append(f"- AML：{int(total['AML_0'])}")
    lines.append(f"- ALL：{int(total['ALL_1'])}")
    lines.append(f"- CML：{int(total['CML_2'])}")
    lines.append(f"- CLL：{int(total['CLL_3'])}")
    lines.append(f"- AML 中 M3/APL：{int(total['AML_M3_1'])}")
    lines.append(f"- AML 中非 M3：{int(total['AML_non_M3_0'])}")
    lines.append("")
    lines.append("按中心/sheet 汇总见 `master_summary.csv` 和 `prepared_data_summary.xlsx`。")
    lines.append("")
    lines.append("## 3. 单位和异常标记")
    lines.append("")
    lines.append("已生成以下标记：")
    lines.append("")
    lines.append("- `HCT_unit_flag`：HCT 转换为百分数后仍 <5 或 >80。")
    lines.append("- `PCT_unit_flag`：PCT <0 或 >5。")
    lines.append("- `FIB_unit_flag`：FIB <=0 或 >20。")
    lines.append("- `diff_percent_flag`：LYM%+MONO%+NEUT%+BASO%+EO% 不在 95-105。")
    lines.append("")
    lines.append("注意：HCT 已按规则标准化为百分数，同时保留 `HCT_raw`。")
    lines.append("")
    lines.append("## 4. 输出文件")
    lines.append("")
    lines.append("- `master_target_labeled.csv`：主分析标准化数据。")
    lines.append("- `excluded_rows_audit.csv`：被排除记录审计表。")
    lines.append("- `standard_feature_mapping_used.csv`：本次使用的字段映射。")
    lines.append("- `master_summary.csv`：样本量和 flag 汇总。")
    lines.append("- `prepared_data_summary.xlsx`：Excel 汇总。")
    lines.append("")
    lines.append("## 5. 下一步")
    lines.append("")
    lines.append("1. 人工确认 `standard_feature_mapping_used.csv` 中 `multiple_used_first` 的映射是否正确，尤其铜陵和深圳的凝血/生化字段。")
    lines.append("2. 根据患者 ID/姓名/日期做去重策略。")
    lines.append("3. 统计 CBC、CBC+Bio、CBC+Coag、CBC+Bio+Coag 的可用样本量。")
    lines.append("4. 再开始模型训练。")
    (OUT_DIR / "prepared_data_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
