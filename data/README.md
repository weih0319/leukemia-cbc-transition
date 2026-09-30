# Data are not included

This repository contains analysis code only. The retrospective patient-level source files cannot be distributed publicly.

To reproduce the analysis with appropriately authorized local data, create `data/raw/` and place the following workbooks there using the original filenames:

| File | Sheets used |
|---|---|
| `1.安医大.xlsx` | `Sheet1` |
| `2.淮北煤炭.xlsx` | `急性髓系`, `急性淋系`, `慢性髓系`, `慢性淋系` |
| `5.铜陵.xlsx` | `AML`, `ALL`, `CML`, `CLL` |

The loader recognizes centre-specific source columns through the mappings in `prepare_master_tables.py`. Required analysis fields are patient linkage variables available to the authorized investigators, test date, recorded sex, age, diagnosis/subtype, WBC, HGB, PLT, absolute NEUT and NEUT percentage.

Units used in the study are `x10^9/L` for WBC, PLT and absolute NEUT and `g/L` for HGB. The analytical thresholds are WBC >=100, HGB <60, PLT <20 and NEUT <0.5 in those units.

Do not commit source workbooks, direct identifiers, linkage keys, deidentified row-level exports, prediction files or serialized models to a public repository. The MIT License applies to code only and does not grant permission to distribute data, manuscript text or institutional documents.
