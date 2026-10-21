#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_download.py
==============
Retrieve raw annual financial-statement files from the Indonesia Stock Exchange
(IDX) disclosure portal. This documents how the source data were obtained; it
is NOT required to reproduce the analysis (the derived panels are already in
../data/). Run it only to rebuild the dataset from scratch.

PROVENANCE / MANUAL STEP
------------------------
The IDX portal is served behind Cloudflare, which blocks non-browser requests
from data-centre IP ranges. The per-year report *listings* (JSON) were therefore
saved manually from a signed, warmed browser session, one file per fiscal year:

  In a browser already on https://www.idx.co.id, open each of these URLs and
  save the raw JSON as idx_<year>.json next to this script (year = 2021..2025):

  https://www.idx.co.id/primary/ListedCompany/GetFinancialReport?indexFrom=0&pageSize=1000&year=<year>&reportType=rdf&EmitenType=s&periode=audit&kodeEmiten=&SortColumn=KodeEmiten&SortOrder=asc

Each JSON lists every issuer's audited annual report and its attachment paths
(the FinancialStatement .xlsx workbook and the instance.zip XBRL archive).
This script reads those listings and downloads the attachments. It uses
curl_cffi, which impersonates a real Chrome TLS fingerprint and clears
Cloudflare from an ordinary (non-data-centre) network such as a home or campus
connection; it will not work from a cloud server.

Outputs:
    downloads/<year>/<CODE>.xlsx          financial-statement workbooks
    downloads/<year>/zip/<CODE>.zip       XBRL archives (fallback for corrupt xlsx)
    download_log.csv                      issuers that failed or had no xlsx

Usage:
    pip install curl_cffi
    python 00_download.py
"""

import os
import json
import csv
import time
import random
from pathlib import Path

try:
    from curl_cffi import requests
    SESSION = requests.Session(impersonate="chrome")
except ImportError:
    raise SystemExit("Install curl_cffi first:  pip install curl_cffi")

YEARS = [2021, 2022, 2023, 2024, 2025]
BASE = "https://www.idx.co.id"
HERE = Path(__file__).resolve().parent
OUT = HERE / "downloads"
DELAY = (2.0, 5.0)          # polite random delay between downloads (seconds)
MAX_RETRY = 3

SESSION.headers.update({"Referer": "https://www.idx.co.id/"})


def pick_attachment(attachments, kind):
    """kind='xlsx' -> FinancialStatement workbook; kind='zip' -> instance.zip."""
    if kind == "xlsx":
        for a in attachments:
            if a.get("File_Type") == ".xlsx" and "FinancialStatement" in a.get("File_Name", ""):
                return a
    else:
        for a in attachments:
            if a.get("File_Name") == "instance.zip":
                return a
    return None


def fetch(url, dest, magic):
    for attempt in range(1, MAX_RETRY + 1):
        try:
            r = SESSION.get(url, timeout=60)
            r.raise_for_status()
            if not r.content.startswith(magic):          # xlsx & zip both start b"PK"
                raise ValueError("unexpected content (blocked page or corrupt file)")
            dest.write_bytes(r.content)
            return True
        except Exception as e:
            time.sleep(attempt * 10)
            last = str(e)
    print(f"      failed: {last[:80]}")
    return False


def main():
    log = []
    for year in YEARS:
        listing = HERE / f"idx_{year}.json"
        if not listing.exists():
            print(f"[!] {listing.name} not found - save it from the browser first (see header).")
            continue
        data = json.loads(listing.read_text(encoding="utf-8-sig"))
        results = data.get("Results", [])
        outdir = OUT / str(year)
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"=== {year}: {len(results)} issuers ===")
        for i, r in enumerate(results, 1):
            code = r.get("KodeEmiten", "?").strip()
            att = pick_attachment(r.get("Attachments", []), "xlsx")
            if att is None:
                log.append([year, code, "NO_XLSX"]); continue
            dest = outdir / f"{code}.xlsx"
            if dest.exists() and dest.stat().st_size > 1000:
                continue
            url = BASE + att["File_Path"].replace(" ", "%20")
            ok = fetch(url, dest, b"PK")
            if not ok:
                log.append([year, code, "FAILED_XLSX"])
            time.sleep(random.uniform(*DELAY))

    # Fallback: fetch instance.zip for issuers whose xlsx failed / was corrupt.
    for year, code, status in list(log):
        if status not in ("FAILED_XLSX",):
            continue
        listing = HERE / f"idx_{year}.json"
        data = json.loads(listing.read_text(encoding="utf-8-sig"))
        rec = next((r for r in data["Results"] if r.get("KodeEmiten", "").strip() == code), None)
        att = pick_attachment(rec.get("Attachments", []), "zip") if rec else None
        if att is None:
            continue
        zdir = OUT / str(year) / "zip"; zdir.mkdir(parents=True, exist_ok=True)
        url = BASE + att["File_Path"].replace(" ", "%20")
        fetch(url, zdir / f"{code}.zip", b"PK")
        time.sleep(random.uniform(*DELAY))

    with open(HERE / "download_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["year", "code", "status"]); w.writerows(log)
    print(f"[i] Done. Problems logged to download_log.csv ({len(log)} entries).")


if __name__ == "__main__":
    main()
