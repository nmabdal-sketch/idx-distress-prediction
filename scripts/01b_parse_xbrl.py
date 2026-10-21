#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01b_parse_xbrl.py
=================
Recover firm-year records from XBRL instance archives for the 39 issuer-years
whose FinancialStatement .xlsx workbooks were corrupted at source (served as
~165-byte files). The downloaded instance.zip archives contain an
`instance.xbrl` document from which the same items are read directly, using the
IDX 2020 taxonomy element names.

Inputs:   downloads/<year>/zip/<CODE>.zip   (each contains instance.xbrl)
Output:   ../data/panel_xbrl.csv            (same columns as panel_xlsx.csv)
Usage:    python 01b_parse_xbrl.py
"""

import csv
import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

YEARS = [2021, 2022, 2023, 2024, 2025]
HERE = Path(__file__).resolve().parent
IN_DIR = HERE / "downloads"
OUT = HERE / ".." / "data" / "panel_xbrl.csv"

COR = "{http://www.idx.co.id/xbrl/taxonomy/2020-01-01/cor}"
DEI = "{http://www.idx.co.id/xbrl/taxonomy/2020-01-01/dei}"

# taxonomy element -> (column, expected context)
NUM = {
    "CurrentAssets": ("aset_lancar", "CurrentYearInstant"),
    "CurrentLiabilities": ("liab_lancar", "CurrentYearInstant"),
    "Assets": ("total_aset", "CurrentYearInstant"),
    "Liabilities": ("total_liab", "CurrentYearInstant"),
    "Equity": ("total_ekuitas", "CurrentYearInstant"),
    "AppropriatedRetainedEarnings": ("saldo_laba_ditentukan", "CurrentYearInstant"),
    "UnappropriatedRetainedEarnings": ("saldo_laba_belum", "CurrentYearInstant"),
    "SalesAndRevenue": ("pendapatan", "CurrentYearDuration"),
    "ProfitLossBeforeIncomeTax": ("laba_sebelum_pajak", "CurrentYearDuration"),
    "InterestAndFinanceCosts": ("beban_bunga", "CurrentYearDuration"),
    "ProfitLoss": ("laba_bersih", "CurrentYearDuration"),
}
DEI_MAP = {
    "EntityName": "nama", "EntityMainIndustry": "industri_utama",
    "Sector": "sektor", "Subsector": "subsektor",
    "DescriptionOfPresentationCurrency": "mata_uang",
    "LevelOfRoundingUsedInFinancialStatements": "pembulatan",
    "TypeOfAuditorsOpinion": "opini_audit",
}


def parse_zip(path, year):
    row = {"kode": path.stem, "tahun": year, "sumber": "xbrl"}
    flags = []
    try:
        with zipfile.ZipFile(path) as z:
            name = next((n for n in z.namelist() if n.lower().endswith(".xbrl")), None)
            if name is None:
                row["flag_masalah"] = "ZIP_TANPA_XBRL"; return row
            tree = ET.parse(io.BytesIO(z.read(name)))
    except Exception as e:
        row["flag_masalah"] = f"GAGAL_BUKA: {e}"; return row

    for el in tree.getroot():
        tag = el.tag
        if tag.startswith(COR):
            nm = tag[len(COR):]
            if nm in NUM:
                col, ctx = NUM[nm]
                if el.get("contextRef") == ctx and el.text:
                    try:
                        row[col] = float(el.text)
                    except ValueError:
                        pass
        elif tag.startswith(DEI):
            nm = tag[len(DEI):]
            if nm in DEI_MAP and el.text:
                row[DEI_MAP[nm]] = el.text.strip()

    _derive(row, flags)
    row["flag_masalah"] = ";".join(flags)
    return row


def _derive(row, flags):
    def g(k):
        v = row.get(k)
        return v if isinstance(v, float) else None
    ta, tl = g("total_aset"), g("total_liab")
    ca, cl = g("aset_lancar"), g("liab_lancar")
    eq = g("total_ekuitas")
    re_total = None
    if g("saldo_laba_ditentukan") is not None or g("saldo_laba_belum") is not None:
        re_total = (g("saldo_laba_ditentukan") or 0) + (g("saldo_laba_belum") or 0)
    lsp, bunga = g("laba_sebelum_pajak"), g("beban_bunga")
    ebit = (lsp + (bunga or 0)) if lsp is not None else None
    row["saldo_laba"] = re_total
    row["working_capital"] = (ca - cl) if (ca is not None and cl is not None) else None
    row["ebit"] = ebit
    if ta and ta != 0:
        x1 = row["working_capital"] / ta if row["working_capital"] is not None else None
        x2 = re_total / ta if re_total is not None else None
        x3 = ebit / ta if ebit is not None else None
        x4 = eq / tl if (eq is not None and tl not in (None, 0)) else None
        row["X1"], row["X2"], row["X3"], row["X4"] = x1, x2, x3, x4
        if None not in (x1, x2, x3, x4):
            row["Z2_score"] = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
        else:
            flags.append("Z_TIDAK_LENGKAP")
        row["ROA"] = g("laba_bersih") / ta if g("laba_bersih") is not None else None
        row["leverage"] = tl / ta if tl is not None else None
        row["current_ratio"] = ca / cl if (ca is not None and cl not in (None, 0)) else None
    else:
        flags.append("TOTAL_ASET_KOSONG")


COLS = ["kode", "tahun", "sumber", "nama", "industri_utama", "sektor", "subsektor",
        "mata_uang", "pembulatan", "opini_audit",
        "aset_lancar", "liab_lancar", "total_aset", "total_liab",
        "saldo_laba", "total_ekuitas", "pendapatan", "laba_sebelum_pajak",
        "beban_bunga", "laba_bersih", "working_capital", "ebit",
        "X1", "X2", "X3", "X4", "Z2_score", "ROA", "leverage",
        "current_ratio", "flag_masalah"]


def main():
    rows = []
    for year in YEARS:
        folder = IN_DIR / str(year) / "zip"
        if not folder.exists():
            continue
        for f in sorted(folder.glob("*.zip")):
            rows.append(parse_zip(f, year))
    if not rows:
        print("[!] No instance.zip files found under downloads/<year>/zip/.")
        return
    with open(OUT, "w", newline="", encoding="utf-8-sig") as fo:
        w = csv.DictWriter(fo, fieldnames=COLS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"[i] {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
