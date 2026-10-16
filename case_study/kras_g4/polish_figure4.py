"""Render the KRAS workflow from existing stage data and plotted assay coordinates."""

from pathlib import Path
import json
import re
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))   # shared figure style lives here
from style import setup_publication_style, save_all, INK, MUTED, GRID

BASE = Path(__file__).resolve().parent
DATA = BASE / "denovo2"   # stage manifests written by run_chain_ber102.py
OUT = ROOT / "reproduction" / "figures"
TEAL = "#246B70"
GREY = "#CDD3D9"


def stage(name):
    return pd.DataFrame(json.loads((DATA / name).read_text()))


def assay_coordinates():
    """Assay points, SDs and the fitted displacement curve for panel g.

    The canonical source is ``kd_fit.svg``, the graph export that this script
    converts from SVG user units. That file is not part of the repository, so when
    it is absent we fall back to the coordinates previously extracted from it
    (``figure4_assay_plotted_values.csv``) and refit the four-parameter logistic so
    the fitted curve is still drawn. The fallback prints which route it used.
    """
    fit_svg = BASE / "kd_fit.svg"
    if not fit_svg.is_file():
        from scipy.optimize import curve_fit
        table = pd.read_csv(BASE / "figure4_assay_plotted_values.csv")
        x = table["concentration_um"].to_numpy(float)
        y = table["normalised_fluorescence"].to_numpy(float)
        sd = table["sd"].to_numpy(float)

        def f4pl(xx, bottom, top, dc50, hill):
            return bottom + (top - bottom) / (1.0 + (xx / dc50) ** hill)

        popt, _ = curve_fit(f4pl, x, y, p0=[y.min(), y.max(), np.median(x), 1.0],
                            maxfev=200000)
        grid = np.logspace(np.log10(x.min()), np.log10(x.max()), 300)
        print(f"  assay: kd_fit.svg absent, refit from CSV -> DC50 = {popt[2]:.2f} uM")
        return np.column_stack((x, y)), sd, np.column_stack((grid, f4pl(grid, *popt)))

    print("  assay: using kd_fit.svg")
    root = ET.parse(fit_svg).getroot()
    namespace = {"s": "http://www.w3.org/2000/svg"}
    def group(ident):
        return root.find(f'.//*[@id="{ident}"]')
    def xy(path):
        values = np.array([float(value) for value in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", path)])
        return values.reshape(-1, 2)
    def convert(points):
        return np.column_stack((10 ** (2 + (points[:, 0] - 113.083807) / 87.603409) / 1000,
                                (305.882439 - points[:, 1]) / 259.226217))
    points = np.array([[float(element.get("x")), float(element.get("y"))]
                       for element in group("line2d_115").findall('.//s:use', namespace)])
    curve = convert(xy(group("line2d_113").find('s:path', namespace).get("d")))
    errors = [convert(xy(element.get("d"))) for element in group("LineCollection_1").findall('s:path', namespace)]
    values = convert(points)
    sd = np.array([abs(pair[1, 1] - pair[0, 1]) / 2 for pair in errors])
    return values, sd, curve


def label(ax, letter):
    ax.text(-0.17, 1.12, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def workflow(ax):
    ax.axis("off")
    phases = [
        ("Generation & fragmentation", "#E9F1F1", [
            ("10,077 molecules", "NA2smol · BER102 pocket (7X8M)"),
            ("9,278 ring fragments", "Fragmentation and deduplication"),
            ("717 fragments retained", "Docking, size and property gate")]),
        ("Fragment-based amplification", "#F4ECE7", [
            ("1,588 variants", "Saturated heteroatom mutation"),
            ("216,092 candidate pairs", "Port-matched recombination"),
            ("9,812 pairs → 1,721 products", "Geometry filter and deduplication")]),
        ("Screening & experimental test", "#ECEBF3", [
            ("785 retained → 728 resolved", "Property gate and relaxed docking"),
            ("293 Tier-1 candidates", "RNAsmol ensemble + rDock ranking"),
            ("34 PubChem records → 14 vendor-listed", "candidates → 1 tested (ThT displacement)")])]
    for column, (title, shade, steps) in enumerate(phases):
        left = column / 3 + 0.012
        width = 0.309
        ax.add_patch(Rectangle((left, 0.02), width, 0.94, facecolor=shade, edgecolor="none"))
        ax.text(left + width / 2, 0.9, title, ha="center", fontsize=7, fontweight="bold")
        for row, (count, operation) in enumerate(steps):
            bottom = 0.65 - row * 0.27
            ax.add_patch(FancyBboxPatch((left + 0.014, bottom), width - 0.028, 0.19,
                        boxstyle="round,pad=0.004,rounding_size=0.015",
                        facecolor="white", edgecolor="#B6C3C8", linewidth=0.65))
            ax.text(left + width / 2, bottom + 0.122, count, ha="center",
                    fontsize=5.8 if len(count) > 36 else 6.6, fontweight="bold")
            ax.text(left + width / 2, bottom + 0.053, operation, ha="center", fontsize=5.8, color=MUTED)
            if row < 2:
                ax.annotate("", (left + width / 2, bottom - 0.072), (left + width / 2, bottom),
                            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": MUTED})
        if column < 2:
            ax.annotate("", (left + width + 0.028, 0.52), (left + width, 0.52),
                        arrowprops={"arrowstyle": "->", "lw": 1.1, "color": INK})
    ax.text(-0.01, 1, "a", fontsize=9, fontweight="bold")


def main():
    setup_publication_style()
    fig = plt.figure(figsize=(7.2, 8.2), facecolor="white")
    workflow(fig.add_axes([0.02, 0.755, 0.97, 0.235]))
    fragment_ax = fig.add_axes([0.065, 0.495, 0.18, 0.20])
    product_ax = fig.add_axes([0.315, 0.495, 0.18, 0.20])
    relaxed_ax = fig.add_axes([0.565, 0.495, 0.18, 0.20])
    votes_ax = fig.add_axes([0.815, 0.495, 0.18, 0.20])
    assay_ax = fig.add_axes([0.745, 0.10, 0.24, 0.25])
    fragments = stage("paper_stage1_fragments.json").dropna(subset=["smina", "QED"])
    products = stage("paper_stage4_products.json").dropna(subset=["QED", "SAS"])
    for ax, frame, xkey, ykey, passkey, title, letter in [
        (fragment_ax, fragments, "smina", "QED", "pass_pool", "Fragment gate\n717 / 9,278 retained", "b"),
        (product_ax, products, "QED", "SAS", "pass_filter", "Product gate\n785 / 1,721 retained", "c")]:
        passed = frame[passkey].astype(bool)
        ax.scatter(frame.loc[~passed, xkey], frame.loc[~passed, ykey], s=3, color=GREY, alpha=0.35, linewidths=0, rasterized=True)
        ax.scatter(frame.loc[passed, xkey], frame.loc[passed, ykey], s=4, color=TEAL, alpha=0.65, linewidths=0, rasterized=True)
        ax.set_title(title, loc="left", fontsize=6.8, pad=5)
        ax.grid(axis="y", color=GRID, linewidth=0.45)
        ax.set_axisbelow(True)
        label(ax, letter)
    votes_ax.text(0.98, 0.04, "Tier-1: teal\nOther: grey", transform=votes_ax.transAxes,
                  ha="right", va="bottom", fontsize=5.5)
    fragment_ax.axvline(-2.5, color=MUTED, ls="--", lw=0.7)
    fragment_ax.axhline(0.5, color=MUTED, ls="--", lw=0.7)
    fragment_ax.set(xlabel="smina (kcal mol$^{-1}$)", ylabel="Fragment QED", ylim=(0.2, 1))
    product_ax.axvline(0.5, color=MUTED, ls="--", lw=0.7)
    product_ax.axhline(2.16, color=MUTED, ls="--", lw=0.7)
    product_ax.set(xlabel="Product QED", ylabel="SA score", ylim=(1.4, 4))
    tiers = stage("chain_tiers.json")
    selected = tiers["tier1"].astype(bool)
    for ax, ykey, title, letter in [(relaxed_ax, "QED", "Relaxed docking", "e"),
                                    (votes_ax, "model_votes", "Ensemble triage", "f")]:
        yvalues = tiers[ykey].to_numpy(dtype=float)
        if ykey == "model_votes":
            yvalues = yvalues + np.random.default_rng(7).uniform(-0.14, 0.14, len(tiers))
        ax.scatter(tiers.loc[~selected, "ms_INTER"], yvalues[~selected], s=4, color=GREY, alpha=0.55, linewidths=0)
        ax.scatter(tiers.loc[selected, "ms_INTER"], yvalues[selected], s=4, color=TEAL, alpha=0.65, linewidths=0)
        ax.axvline(tiers["ms_INTER"].median(), color=MUTED, ls="--", lw=0.7)
        ax.axhline(0.5 if ykey == "QED" else 3, color=MUTED, ls="--", lw=0.7)
        ax.set_xlabel("rDock score")
        ax.set_ylabel("QED" if ykey == "QED" else "RNAsmol votes")
        ax.set_title(title + "\n293 / 728 Tier-1", loc="left", fontsize=6.8)
        if ykey == "model_votes":
            ax.set_yticks([1, 2, 3, 4, 5])
        label(ax, letter)
    values, sd, curve = assay_coordinates()
    pd.DataFrame({"concentration_um": values[:, 0], "normalised_fluorescence": values[:, 1], "sd": sd}).to_csv(OUT / "figure4_assay_plotted_values.csv", index=False)
    assay_ax.plot(curve[:, 0], curve[:, 1], color=TEAL, lw=1.3)
    assay_ax.errorbar(values[:, 0], values[:, 1], yerr=sd, fmt="o", color=TEAL, markersize=2.8,
                      ecolor=MUTED, elinewidth=0.65, capsize=1.5)
    assay_ax.axvline(22.387, color=MUTED, ls="--", lw=0.7)
    assay_ax.set(xscale="log", xlabel="Concentration (µM)", ylabel="Normalised ThT fluorescence", ylim=(-0.05, 1.08))
    assay_ax.set_title("ThT displacement", loc="left", fontsize=6.8)
    assay_ax.text(0.97, 0.96, "DC$_{50}$ = 22.4 µM\nMean ± SD; 3 wells", transform=assay_ax.transAxes,
                  ha="right", va="top", fontsize=6)
    label(assay_ax, "g")
    registered = [record for record in json.loads((DATA / "tier1_pubchem.json").read_text()) if record.get("cid")]
    metrics = {record["smiles"]: record for record in tiers.to_dict("records")}
    for record in registered:
        record["model_votes"] = metrics[record["smiles"]]["model_votes"]
        record["QED"] = metrics[record["smiles"]]["QED"]
        record["SAS"] = metrics[record["smiles"]]["SAS"]
    registered.sort(key=lambda record: (-record["model_votes"], record["ms_INTER"], record["cid"]))
    fig.text(0.02, 0.405, "d", fontweight="bold", fontsize=9)
    fig.text(0.06, 0.405, "34 PubChem candidates", fontsize=7)
    fig.text(0.06, 0.389, "Order: votes ↓, rDock ↑ · Labels: number · CID · votes/5 · * vendor listed", fontsize=5.7)
    columns = 7
    for index, record in enumerate(registered):
        row, column = divmod(index, columns)
        left = 0.025 + column * 0.091
        bottom = 0.318 - row * 0.068
        if record["cid"] == 74755:
            fig.add_artist(Rectangle((left - 0.002, bottom - 0.022), 0.091, 0.078,
                                     transform=fig.transFigure, facecolor="none",
                                     edgecolor=TEAL, linewidth=0.9))
        ax = fig.add_axes([left, bottom, 0.087, 0.055])
        molecule = Chem.MolFromSmiles(record["smiles"])
        ax.imshow(Draw.MolToImage(molecule, size=(450, 250)), interpolation="none")
        ax.axis("off")
        vendor = "*" if record.get("has_vendors") else ""
        fig.text(left + 0.0435, bottom - 0.002, f'{index + 1:02d} · {record["cid"]}{vendor}\n{record["model_votes"]}/5 votes',
                 ha="center", va="top", fontsize=4.8)
    pd.DataFrame([{ "candidate": index + 1, **record} for index, record in enumerate(registered)]).to_csv(
        OUT / "figure4_candidate_labels.csv", index=False)
    save_all(fig, OUT / "figure4", include_tiff=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
