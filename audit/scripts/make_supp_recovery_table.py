#!/usr/bin/env python
"""Emit the full row-level recovery-sensitivity Supplementary table (LaTeX)."""
from pathlib import Path

import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
ROOT = Path(__file__).resolve().parents[2]

def rounded(value, signed=False):
    q = Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    if q == 0:
        q = abs(q)
    return format(q, "+.3f" if signed else ".3f")

AUDIT = (ROOT / "audit")
OUT = (ROOT / "reproduction" / "results" / "si_supp_recovery_full.tex")

FLAB = {"ft_exact": "exact FT", "ft_T0.5": "FT $\\geq$0.5", "ft_T0.6": "FT $\\geq$0.6",
        "ft_T0.7": "FT $\\geq$0.7", "pretrain_exact": "exact PT", "pretrain_T0.5": "PT $\\geq$0.5"}
ORDER = ["ft_exact", "ft_T0.5", "ft_T0.6", "ft_T0.7", "pretrain_exact", "pretrain_T0.5"]
NAME = {"RNA2smol": "NA2smol", "MolCRAFT": "MolCRAFT", "Pocket2Mol": "Pocket2Mol",
        "ED2Mol": "ED2Mol", "ablation:RNA2smol (Baseline)": "no-pretraining baseline",
        "ablation:Pretrain Only": "Pretrain Only",
        "ablation:Pretrain+Constrain": "Pretrain+Constrain"}


def fmt(v, lo, hi):
    return f"${rounded(v, True)}$ [${rounded(lo, True)}$, ${rounded(hi, True)}$]"


def panel(df, title):
    lines = [f"\\textbf{{{title}}}", "", "\\setlength{\\tabcolsep}{3pt}", "\\scriptsize",
             "\\begin{tabular}{@{}llrrr@{}}", "\\toprule",
             "\\textbf{Method} & \\textbf{Filter} & \\textbf{n} & \\textbf{$A_{\\text{orig}}\\to A_{\\text{filt}}$} & \\textbf{$\\Delta A$ (95\\% CI)} \\\\",
             "\\midrule"]
    for _, r in df.iterrows():
        lines.append(f"{NAME.get(r['method'], r['method'])} & {FLAB[r['filter']]} & "
                     f"{int(r['n_eligible'])} & ${rounded(r['A_original_subset'])}\\to {rounded(r['A_filtered_subset'])}$ & "
                     f"{fmt(r['delta_filter_effect'], r['delta_ci_lo'], r['delta_ci_hi'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return lines


def main():
    fix = pd.read_csv(AUDIT / "tables" / "recovery_fixed_target_sensitivity.csv")
    abl = pd.read_csv(AUDIT / "tables" / "recovery_ablation_ladder.csv")
    paired = pd.read_csv(AUDIT / "tables" / "recovery_ablation_paired_ci.csv")
    pci = paired[paired.kind == "delta_A"].set_index(["filter", "comparison"])
    # Use the full-model paired ablation run for duplicated SMRT-Net cells.
    for _, row in abl[(abl.config == "RNA2smol") & abl["filter"].isin(ORDER)].iterrows():
        mask = (fix.benchmark == "SMRT-Net") & (fix.method == "RNA2smol") & (fix["filter"] == row["filter"])
        assert (fix.loc[mask, "n_eligible"] == row.n_eligible).all()
        for col in ["A_original_subset", "A_filtered_subset"]:
            fix.loc[mask, col] = row[col]
        fix.loc[mask, "delta_filter_effect"] = row.delta
        fix.loc[mask, "delta_ci_lo"] = pci.loc[(row["filter"], "RNA2smol"), "ci_lo"]
        fix.loc[mask, "delta_ci_hi"] = pci.loc[(row["filter"], "RNA2smol"), "ci_hi"]

    L = ["\\section*{Supplementary Table 2 | Row-level recovery sensitivity}",
         "",
         "\\noindent For every method/configuration and reference filter, recovery is recomputed on the "
         "common eligible-target subset that survives the filter ($A_{\\text{orig}}$, unfiltered references; "
         "$A_{\\text{filt}}$, filtered references). $\\Delta A = A_{\\text{filt}}-A_{\\text{orig}}$ with the "
         "paired 95\\% target-bootstrap interval. FT = fine-tuning training partition; PT = ligand-only "
         "pretraining library; $\\geq$0.5/0.6/0.7 = maximum ECFP4-Tanimoto similarity threshold.",
         ""]
    for i, bench in enumerate(["SMRT-Net", "DUD-E"], 1):
        d = fix[(fix.benchmark == bench) & (fix.method.isin(["RNA2smol", "MolCRAFT", "Pocket2Mol", "ED2Mol"]))]
        d = d[d["filter"].isin(ORDER)]
        d = d.assign(_o=d["filter"].map({f: i for i, f in enumerate(ORDER)})).sort_values(["_o", "method"])
        L += panel(d, f"Panel A{i}. {bench}: NA2smol and external generators")
        L.append("")
    rows = []
    for _, r in abl.iterrows():
        if r["filter"] not in ORDER:
            continue
        c = r["config"]
        lo = pci.loc[(r["filter"], c), "ci_lo"] if (r["filter"], c) in pci.index else float("nan")
        hi = pci.loc[(r["filter"], c), "ci_hi"] if (r["filter"], c) in pci.index else float("nan")
        rows.append(dict(method=c, filter=r["filter"], n_eligible=r["n_eligible"],
                         A_original_subset=r["A_original_subset"],
                         A_filtered_subset=r["A_filtered_subset"],
                         delta_filter_effect=r["delta"], delta_ci_lo=lo, delta_ci_hi=hi))
    dd = pd.DataFrame(rows)
    pd.concat([fix.assign(panel="external"), dd.assign(benchmark="SMRT-Net", panel="ablation")], ignore_index=True).to_csv(AUDIT / "tables" / "recovery_manuscript_authoritative.csv", index=False)
    dd["_o"] = dd["filter"].map({f: i for i, f in enumerate(ORDER)})
    dd = dd.sort_values(["_o", "method"])
    L += panel(dd, "Panel B. SMRT-Net sequential ablation configurations")
    L.append("\\noindent{\\scriptsize All $\\Delta A$ values are filtered-minus-original changes on identical "
             "eligible targets. $A_{\\text{orig}}$ here is the unfiltered-reference recovery on the "
             "filter-matched subset, not the all-target value.}")
    OUT.write_text("\n".join(L) + "\n")
    print("wrote", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
