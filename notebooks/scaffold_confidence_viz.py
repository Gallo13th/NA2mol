#!/usr/bin/env python
"""Example molecule visualization: BM scaffold vs substituent confidence."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
from PIL import Image

OUT_DIR = Path(__file__).resolve().parent / 'output'
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.sans-serif': ['Arial', 'Helvetica'], 'font.family': 'sans-serif',
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300,
})

# ── Load example molecule ────────────────────────────────────────────────────
SDF_PATH = (ROOT / 'notebooks' / 'output' / 'dude_onlygeneral_0506'
                '/dude_h_1c8k/dude_h_1c8k_conf.sdf')
suppl = Chem.SDMolSupplier(str(SDF_PATH))
mol = next(suppl)

atom_confs = np.array([float(x) for x in mol.GetProp('_AtomConfidenceList').split(',')])
mol_conf = float(mol.GetProp('_MolConfidence'))
affinity = float(mol.GetProp('minimizedAffinity'))
rmsd = float(mol.GetProp('minimizedRMSD'))

# Compute scaffold
scaff_mol = MurckoScaffold.GetScaffoldForMol(mol)
match = mol.GetSubstructMatch(scaff_mol)
is_scaff = np.zeros(mol.GetNumAtoms(), dtype=bool)
is_scaff[list(match)] = True

scaff_conf_mean = atom_confs[is_scaff].mean()
subs_conf_mean = atom_confs[~is_scaff].mean()
n_scaff = is_scaff.sum()
n_subs = (~is_scaff).sum()

print(f"Molecule: {Chem.MolToSmiles(mol)}")
print(f"Scaffold: {Chem.MolToSmiles(scaff_mol)}")
print(f"Scaffold (n={n_scaff}): mean conf = {scaff_conf_mean:.4f}")
print(f"Substituent (n={n_subs}): mean conf = {subs_conf_mean:.4f}")
print(f"Δ = {scaff_conf_mean - subs_conf_mean:+.4f}")
print(f"_MolConfidence = {mol_conf:.4f}, affinity = {affinity:.2f}, RMSD = {rmsd:.2f}")

# ── Helper: draw molecule with atom coloring and confidence annotations ──────
def draw_mol_with_confidence(mol, atom_confs, highlight_atoms, highlight_color,
                              title, atom_labels=None):
    """Draw molecule with atoms colored by confidence and optional highlights."""
    n = mol.GetNumAtoms()
    # Normalize confidence to 0-1 for colormap
    vmin, vmax = 0.05, 0.55

    # Build atom colors (confidence colormap: RdYlGn, low=red, high=green)
    atom_colors = {}
    atom_radii = {}
    for i in range(n):
        # Clipped and normalized confidence
        c = np.clip((atom_confs[i] - vmin) / (vmax - vmin), 0, 1)
        # Blue (low) → Red (high) colormap
        r = c
        g = 0.3 + 0.4 * (1 - abs(c - 0.5) * 2)
        b = 1 - c
        atom_colors[i] = (r, g, b)

    # Build highlight dict
    highlight_atom_colors = {}
    if highlight_atoms is not None:
        for idx in highlight_atoms:
            highlight_atom_colors[idx] = highlight_color

    # Use MolDraw2D for better annotation control
    d2d = rdMolDraw2D.MolDraw2DCairo(600, 450)
    d2d.SetFontSize(0.6)
    opts = d2d.drawOptions()
    opts.addAtomIndices = False
    opts.bondLineWidth = 2.5
    opts.atomLabelFontSize = 14

    # Draw the molecule
    if highlight_atoms is not None:
        d2d.DrawMolecule(mol, highlightAtoms=highlight_atoms,
                         highlightAtomColors=highlight_atom_colors,
                         highlightBondMultipleBonds=True,
                         legend=title)
    else:
        d2d.DrawMolecule(mol, legend=title)
    d2d.FinishDrawing()
    png_data = d2d.GetDrawingText()
    img = Image.open(BytesIO(png_data))
    return img


def draw_mol_confidence_annotated(mol, atom_confs, is_scaff, title):
    """Draw 2D molecule with per-atom confidence labels and scaffold highlighting."""
    n = mol.GetNumAtoms()

    # Conformer for 2D coords
    AllChem.Compute2DCoords(mol)

    d2d = rdMolDraw2D.MolDraw2DCairo(800, 550)
    opts = d2d.drawOptions()
    opts.atomLabelFontSize = 16
    opts.bondLineWidth = 3.0
    opts.addStereoAnnotation = False
    opts.addAtomIndices = False
    opts.highlightRadius = 0.45
    opts.highlightBondWidthMultiplier = 1.5

    # Per-atom colors: scaffold=blue tones, substituent=orange tones
    # But modulated by confidence
    highlight_atoms_scaff = [int(i) for i in np.where(is_scaff)[0]]
    highlight_atoms_subs = [int(i) for i in np.where(~is_scaff)[0]]

    # Draw with dual highlights
    # First draw all atoms
    d2d.DrawMolecule(mol, legend=title)

    d2d.FinishDrawing()
    png_data = d2d.GetDrawingText()
    img = Image.open(BytesIO(png_data))
    return img


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: 3-panel visualization (structure + bar chart + distribution)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating visualization...")

scaff_atoms_list = [int(i) for i in np.where(is_scaff)[0]]
subs_atoms_list = [int(i) for i in np.where(~is_scaff)[0]]
scaff_vals = atom_confs[is_scaff]
subs_vals = atom_confs[~is_scaff]

fig = plt.figure(figsize=(16, 8))

# ── Panel (a): 2D structure, scaffold/subs-colored, per-atom confidence ──────
ax_a = fig.add_axes([0.01, 0.05, 0.52, 0.90])
ax_a.axis('off')

AllChem.Compute2DCoords(mol)

d2d = rdMolDraw2D.MolDraw2DCairo(900, 650)
opts = d2d.drawOptions()
opts.atomLabelFontSize = 16
opts.bondLineWidth = 3.2
opts.addStereoAnnotation = False
opts.addAtomIndices = True
opts.atomLabelMinFontSize = 12

# Per-atom colors: confidence-mapped, with scaffold/subs distinction by ring
atom_cols = {}
atom_rads = {}
for i in range(mol.GetNumAtoms()):
    c_norm = np.clip((atom_confs[i] - 0.05) / 0.5, 0, 1)
    if is_scaff[i]:
        # blue: darker=higher confidence
        atom_cols[i] = (0.15, 0.45 + 0.3*c_norm, 0.6 + 0.35*c_norm)
        atom_rads[i] = 0.45
    else:
        # orange: darker=higher confidence
        atom_cols[i] = (0.9, 0.45 + 0.25*c_norm, 0.1)
        atom_rads[i] = 0.33

d2d.DrawMolecule(
    mol,
    highlightAtoms=scaff_atoms_list + subs_atoms_list,
    highlightAtomColors={int(i): atom_cols[i] for i in range(mol.GetNumAtoms())},
    highlightAtomRadii={int(i): atom_rads[i] for i in range(mol.GetNumAtoms())},
    legend=(f'Scaffold (blue, n={n_scaff}): mean conf = {scaff_conf_mean:.3f}\n'
            f'Substituent (orange, n={n_subs}): mean conf = {subs_conf_mean:.3f}\n'
            f'Δ = {scaff_conf_mean - subs_conf_mean:+.3f}')
)
d2d.FinishDrawing()
img_a = Image.open(BytesIO(d2d.GetDrawingText()))
ax_a.imshow(img_a)

# ── Panel (b): Per-atom confidence bar chart ──────────────────────────────────
ax_b = fig.add_axes([0.56, 0.52, 0.42, 0.43])

n_atoms = mol.GetNumAtoms()
x = np.arange(n_atoms)
bar_colors = ['#2196F3' if is_scaff[i] else '#FF9800' for i in range(n_atoms)]
bar_edges = ['#1565C0' if is_scaff[i] else '#E65100' for i in range(n_atoms)]

ax_b.bar(x, atom_confs, color=bar_colors, edgecolor=bar_edges, linewidth=0.8, alpha=0.9)

ax_b.axhline(y=scaff_conf_mean, color='#1565C0', linestyle='--', linewidth=2,
             label=f'scaffold mean = {scaff_conf_mean:.3f}')
ax_b.axhline(y=subs_conf_mean, color='#E65100', linestyle='--', linewidth=2,
             label=f'substituent mean = {subs_conf_mean:.3f}')

# Element label on top of each bar
for i in range(n_atoms):
    elem = mol.GetAtomWithIdx(i).GetSymbol()
    ax_b.text(i, atom_confs[i] + 0.01, elem, ha='center', fontsize=9,
              fontweight='bold', color='#333333')

# Atom index below
for i in range(n_atoms):
    ax_b.text(i, -0.015, str(i), ha='center', fontsize=7, color='#888888')

ax_b.set_xlim(-0.6, n_atoms - 0.4)
ax_b.set_ylim(0, 0.58)
ax_b.set_xticks([])
ax_b.set_ylabel('Atom confidence', fontsize=11)
ax_b.legend(fontsize=9, frameon=True, edgecolor='#cccccc', loc='upper right')
ax_b.grid(axis='y', alpha=0.12, linestyle='--')
ax_b.set_title(f'(b) Per-atom confidence', fontsize=11, fontweight='bold', loc='left')

# ── Panel (c): KDE distribution comparison ────────────────────────────────────
ax_c = fig.add_axes([0.56, 0.05, 0.42, 0.38])

from scipy.stats import gaussian_kde
for vals, label, color, fill_color in [
    (subs_vals, f'Substituent (n={n_subs})', '#E65100', '#FF9800'),
    (scaff_vals, f'Scaffold (n={n_scaff})', '#1565C0', '#2196F3'),
]:
    if len(vals) >= 3:
        kde = gaussian_kde(vals, bw_method=0.18)
        x_kde = np.linspace(0.03, 0.58, 250)
        y_kde = kde(x_kde)
        ax_c.fill_between(x_kde, 0, y_kde, color=fill_color, alpha=0.35,
                          edgecolor=color, linewidth=1.8)
        ax_c.plot(x_kde, y_kde, color=color, linewidth=2.5, label=label)
        # Rug
        ax_c.plot(vals, np.full_like(vals, -0.3), '|', color=color, markersize=7, alpha=0.7)

ax_c.axvline(scaff_conf_mean, color='#1565C0', linestyle=(0, (4, 3)), linewidth=1.2, alpha=0.7)
ax_c.axvline(subs_conf_mean, color='#E65100', linestyle=(0, (4, 3)), linewidth=1.2, alpha=0.7)

ax_c.set_xlabel('Atom confidence', fontsize=11)
ax_c.set_ylabel('Density', fontsize=11)
ax_c.set_title('(c) Confidence distribution', fontsize=11, fontweight='bold', loc='left')
ax_c.legend(fontsize=9, frameon=True, edgecolor='#cccccc')
ax_c.grid(axis='y', alpha=0.12, linestyle='--')
ax_c.set_ylim(bottom=-0.6)

# MWU annotation
from scipy.stats import mannwhitneyu
ROOT = Path(__file__).resolve().parents[1]
u, p = mannwhitneyu(scaff_vals, subs_vals)
ax_c.text(0.97, 0.94, f'MWU p = {p:.3f}', transform=ax_c.transAxes,
          fontsize=8, ha='right', va='top', style='italic',
          bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc', alpha=0.85))

# Global title
fig.suptitle(f'BM Scaffold vs Substituent Confidence — {Chem.MolToSmiles(mol)}\n'
             f'Affinity = {affinity:.1f} kcal/mol  |  RMSD = {rmsd:.2f} Å  |  '
             f'_MolConfidence = {mol_conf:.3f}',
             fontsize=12, fontweight='bold', y=1.01)

plt.savefig(OUT_DIR / 'scaffold_subs_confidence_example.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {OUT_DIR / 'scaffold_subs_confidence_example.png'}")
print("\nDone.")
