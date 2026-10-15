#!/usr/bin/env python
"""Summarise reference_active_overlap.csv at ligand and target level (with
percentages) for the audit report. Overwrites reference_overlap_summary.csv.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lka_common import AUDIT

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
SOURCES = ["ft_train", "ft_train_protein", "ft_train_rna", "ft_val",
           "ft_unused", "ft_dude_dropped", "pretrain"]


def main():
    o = pd.read_csv(AUDIT / "tables" / "reference_active_overlap.csv")
    rows = []
    for bench, g in o.groupby("benchmark"):
        lig = g.drop_duplicates("canonical")
        n_lig = len(lig)
        n_tgt = g["target_id"].nunique()
        for s in SOURCES:
            rows.append(dict(benchmark=bench, source=s, level="ligand", n=n_lig,
                             exact=int(lig[f"exact_{s}"].sum()),
                             exact_pct=round(100 * lig[f"exact_{s}"].mean(), 2),
                             parent=int(lig[f"parent_{s}"].sum()),
                             parent_pct=round(100 * lig[f"parent_{s}"].mean(), 2),
                             scaffold=int(lig[f"scaffold_{s}"].sum()),
                             scaffold_pct=round(100 * lig[f"scaffold_{s}"].mean(), 2)))
            tg = g.groupby("target_id")
            rows.append(dict(benchmark=bench, source=s, level="target", n=n_tgt,
                             exact=int(tg[f"exact_{s}"].any().sum()),
                             exact_pct=round(100 * tg[f"exact_{s}"].any().mean(), 2),
                             parent=int(tg[f"parent_{s}"].any().sum()),
                             parent_pct=round(100 * tg[f"parent_{s}"].any().mean(), 2),
                             scaffold=int(tg[f"scaffold_{s}"].any().sum()),
                             scaffold_pct=round(100 * tg[f"scaffold_{s}"].any().mean(), 2)))
            for th in THRESHOLDS:
                col = f"maxts_{s}"
                k = int((lig[col] >= th).sum())
                rows.append(dict(benchmark=bench, source=s, level=f"ligand_ts>={th}", n=n_lig,
                                 exact=k, exact_pct=round(100 * k / n_lig, 2)))
                kt = int(g.groupby("target_id")[col].max().ge(th).sum())
                rows.append(dict(benchmark=bench, source=s, level=f"target_ts>={th}", n=n_tgt,
                                 exact=kt, exact_pct=round(100 * kt / n_tgt, 2)))
    out = pd.DataFrame(rows)
    out.to_csv(AUDIT / "tables" / "reference_overlap_summary.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
