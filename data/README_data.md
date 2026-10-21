# Data directory

## Included (derived data — safe to share)

- **panel_xlsx.csv** — firm-year financial-statement items and ratios parsed
  from the standardized XBRL-derived `.xlsx` workbooks published by the IDX.
- **panel_xbrl.csv** — the same fields for 39 issuer-years whose `.xlsx`
  workbooks were corrupted at source, recovered from their XBRL
  `instance.xbrl` archives.

Each row is one issuer-year and contains: issuer code, fiscal year, sector
fields, the raw balance-sheet / income-statement items used, the four Altman
Z"-Score components (X1-X4), the additional ratios (ROA, leverage, current
ratio), the computed Z"-Score, and a data-quality flag column.

These files contain only ratios and totals computed from **publicly filed,
audited financial statements**; they are not proprietary IDX data products.

## NOT included (raw source data)

The ~4,000 underlying `FinancialStatement-<year>-Tahunan-<code>.xlsx` workbooks
and `instance.zip` XBRL archives are the property of the Indonesia Stock
Exchange and are **publicly available** from its official disclosure portal:

    https://www.idx.co.id  ->  Perusahaan Tercatat  ->  Laporan Keuangan dan Tahunan

They are not redistributed in this repository. `scripts/00_download.py` and
`scripts/01*_parse_*.py` document how they were retrieved and converted into
the two derived files above.
