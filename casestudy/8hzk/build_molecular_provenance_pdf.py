"""Generate PDF: molecular-level reproduction provenance chain for mol4 & mol2.

Reproduction standard (revised): stops at the **scaffold-level** end point
achieved by the model + legal post-processing (element substitution,
docking, fixed-scaffold iteration, retaining model-generated substituents).

Grafting (cut model's original substituent + attach a target group such as
N(CCO)C or rhodanine) is judged ILLEGAL (pasting the target answer by hand)
and has been removed from all reproduction paths. Scripts and products that
relied on grafting are archived under casestudy/8hzk/_deprecated_graft/.

Each node = a 2D molecule structure (RDKit) + role + canonical SMILES + Tc +
operation note. Arrows between nodes carry the operation label
(cut / replace). The final node of each chain is highlighted (green bg) and
marked "= scaffold-level reproduction (legal endpoint)". A footer at the end
of each chain states "model reproduction boundary stops here; further
precision is human molecular editing".

Output: casestudy/8hzk/molecular_provenance.pdf
"""
import os
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread
from matplotlib.patches import FancyArrowPatch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

# ---- register CJK fonts (so Chinese glyphs render in the PDF) ----
_CJK_DIR = "/usr/share/fonts/google-noto-cjk"
for _f in ("NotoSansCJK-Regular.ttc", "NotoSansCJK-Bold.ttc"):
    _p = os.path.join(_CJK_DIR, _f)
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

CASE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(CASE, "molecular_provenance.pdf")

MOL4_REF = "CN(CCO)c1cccc(/C=C(\\C#N)c2ccc(-c3ccccc3)cc2)c1"
MOL2_REF = "CN1C(=O)/C(=C/c2cc3ccc(N(C)CCO)cc3s2)SC1=S"

# Legal endpoint markers (final node of each chain).
# `chain_footer` is rendered below the final node, stating the legal boundary.
MOL4_CHAIN = [
    {
        "idx": 1,
        "role": "起点 A · 外部受体种子 (FluoDB 氰基丙烯酸 / 对接)",
        "smi": "C/C=C(\\C#N)C(=O)O",
        "tc": 0.154,
        "op_in": None,
        "note": "对接到 8HZK 口袋 (ccn_seeds_docked.sdf),\n"
                "获得 C=C(C#N) 受体种子。",
        "final": False,
    },
    {
        "idx": 2,
        "role": "起点 B · 模型生成 (dummy_ligand)",
        "smi": "CCC(=O)Nc1cccc(/C=C/c2ccccc2-c2ccc(O)[n+](C)c2)c1",
        "tc": 0.208,
        "op_in": None,
        "note": "限制原子数 20-28 后模型生成。含\n"
                "二苯乙烯桥 + 胺供体模式。",
        "final": False,
    },
    {
        "idx": 3,
        "role": "首批含骨架 (sample_by_seed 汇聚产物)",
        "smi": "CN(C)c1ccc(-c2ccc(C(C#N)=Cc3ccc(C(N)O)cc3N3CCCCC3)cc2)cc1",
        "tc": 0.329,
        "op_in": "汇聚: sample_by_seed\n(固定 CCN 种子 + 模型补全)",
        "note": "首次出现 mol4 的 D-π-A 核心骨架\n"
                "(氰基二苯乙烯), 但带多余装饰\n"
                "(哌啶 / 羟胺)。 [mol4_skeleton_hits #0]",
        "final": False,
    },
    {
        "idx": 4,
        "role": "纯骨架 (切割装饰)",
        "smi": "N#CC(=Cc1cccc(N)c1)c1ccccc1",
        "tc": 0.471,
        "op_in": "切割装饰: SMARTS 定位骨架,\n切去哌啶 / 羟基",
        "note": "纯氰基二苯乙烯 + 芳胺 (NH2) 骨架。\n"
                "[spec07 seeds_round0]",
        "final": False,
    },
    {
        "idx": 5,
        "role": "合法终点 · 迭代产物 (Murcko == mol4)",
        "smi": "N#CC(=Cc1cccc(N)c1)c1ccc(-c2ccccc2)cc1",
        "tc": 0.580,
        "op_in": "固定骨架迭代:\nfixed_atom_mask + sample_by_seed\n3-5 轮",
        "note": "Murcko 骨架精确复现 mol4 (联苯扩展),\n"
                "供体为模型生成的芳胺 NH2 (保留)。\n"
                "[ == 骨架级复现 · 合法终点 ]",
        "final": True,
    },
]

MOL2_CHAIN = [
    {
        "idx": 1,
        "role": "起点 · dummy_ligand #5133 (含吲哚)",
        "smi": "C=C(C)Nc1ccc(-c2cc3ccc(N4CCC[C@@H](O)C4)cc3[nH]2)cc1",
        "tc": 0.149,
        "op_in": None,
        "note": "模型生成的吲哚分子 (限制原子数后)。\n"
                "吲哚 [nH] 是苯并噻吩的等电子体。",
        "final": False,
    },
    {
        "idx": 2,
        "role": "合法终点 · 苯并噻吩骨架 (元素替换)",
        "smi": "C=C(C)Nc1ccc(-c2cc3ccc(N4CCC[C@@H](O)C4)cc3s2)cc1",
        "tc": None,
        "op_in": "元素替换:\natom.SetAtomicNum(16)  NH → S",
        "note": "苯并噻吩骨架 (含 1S), 取代模式与\n"
                "mol2 同构 (供体位/受体位对应)。\n"
                "取代基为模型原产物自带, 保留。\n"
                "[step1_benzothiophene #0]\n"
                "[ == 骨架级复现 · 合法终点 ]",
        "final": True,
    },
]

# Footer text shown at the end of each chain (legal boundary statement).
MOL4_FOOTER = (
    "模型复现边界至此 (节点5, 迭代产物 Tc=0.58, Murcko==mol4)。\n"
    "进一步把芳胺 NH2 替换为 N(CCO)C (目标供体) 属人工分子编辑\n"
    "(切除模型原有取代基 + 接目标基团 = 嫁接, 不合法, 已删除)。\n"
    "→ 该步产物此前声称的 Tc=1.0 不计入模型复现能力。"
)

MOL2_FOOTER = (
    "模型复现边界至此 (节点2, 元素替换苯并噻吩骨架)。\n"
    "进一步引入绕丹宁 / 把哌啶-OH 替换为 N(CCO)C 属人工分子编辑\n"
    "(切除模型原有取代基 + 接目标基团 = 嫁接, 不合法, 已删除)。\n"
    "→ 该步产物此前声称的 Tc=1.0 不计入模型复现能力。"
)


# ----------------------------------------------------------------graphics
def mol_to_image(smi, size=(560, 380), highlight=False):
    """Render a 2D molecule structure to a numpy image array."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"SMILES parse failed: {smi}")
    mc = Chem.Mol(mol.ToBinary())
    try:
        AllChem.Compute2DCoords(mc)
    except Exception:
        pass
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2
    opts.padding = 0.10
    opts.baseFontSize = 0.7
    if highlight:
        opts.setBackgroundColour((0.86, 1.0, 0.86))
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    return imread(BytesIO(png))


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else smi


def tc_str(node):
    return f'Tc = {node["tc"]:.4f}' if node["tc"] is not None else "Tc = —"


def draw_overlay(fig):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def draw_arrow(ax_ov, x, y_top, y_bot, label, ha="left", dx=0.018,
               color="#666666"):
    ax_ov.add_patch(FancyArrowPatch(
        (x, y_top), (x, y_bot),
        arrowstyle="-|>", mutation_scale=22,
        color=color, lw=2.0))
    if label:
        ax_ov.text(x + dx, (y_top + y_bot) / 2, label,
                   fontsize=8.5, va="center", ha=ha, color="#222222",
                   bbox=dict(boxstyle="round,pad=0.35",
                             fc="#fff7d6", ec="#e0c060"))


def node_title(fig, node, x_center, y, va="bottom"):
    is_final = node.get("final", False)
    color = (0.0, 0.45, 0.0) if is_final else "black"
    suffix = "   [ == 骨架级复现 · 合法终点 ]" if is_final else ""
    fig.text(x_center, y,
             f'节点{node["idx"]}  ·  {tc_str(node)}{suffix}',
             ha="center", va=va, fontsize=10.5, fontweight="bold",
             color=color)


def node_text_panel(fig, node, x, y_top, va="top"):
    body = (f'{node["role"]}\n\n'
            f'SMILES (canonical):\n{canon(node["smi"])}\n\n'
            f'说明:\n{node["note"]}')
    fig.text(x, y_top, body,
             fontsize=8.5, va=va, ha="left",
             bbox=dict(boxstyle="round,pad=0.5",
                       fc="#f7f7f7", ec="#cccccc"))


def draw_chain_footer(fig, footer, y=0.025):
    fig.text(0.5, y, footer,
             ha="center", va="bottom", fontsize=9, color="#222222",
             bbox=dict(boxstyle="round,pad=0.5",
                       fc="#fff0f0", ec="#e08080"))


# ----------------------------------------------------------------pages
def draw_cover(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("mol4 & mol2  分子级复现前驱链 (骨架级, 合法)",
                 fontsize=18, fontweight="bold", y=0.95)
    fig.text(0.5, 0.910,
             "Molecular-level Reproduction Provenance  ·  "
             "8HZK case  ·  scaffold-level (legal endpoints only)",
             ha="center", fontsize=10, color="gray")

    overview = (
        "本报告记录两个 known_mol (mol4, mol2) 从具体起始分子, 经过明确的\n"
        "「切割 / 元素替换 / 迭代」操作, 一步步演变到合法骨架级终点的完整路径。\n"
        "每一步标注 SMILES、操作、Tc (Tanimoto vs known_mol)。\n\n"
        "复现标准 (重新界定)\n"
        "   模型 + 合法后处理的复现边界 = 骨架级相近 (Murcko 相同 / 高 Tc)。\n"
        "   从骨架到精确 known_mol 的最后一步属人工分子编辑, 不计入模型复现。\n\n"
        "mol4 链   (5 节点 · 3 步 · 生成导向)\n"
        "   FluoDB CCN 种子 + 模型生成 stilbene (两来源汇聚)\n"
        "   → 切割装饰 → 固定骨架迭代  [ == 骨架级 Tc=0.58, Murcko==mol4 ]\n\n"
        "mol2 链   (2 节点 · 1 步 · 编辑导向)\n"
        "   dummy_ligand #5133 (吲哚) → 元素替换 NH→S\n"
        "   [ == 苯并噻吩骨架级复现, 取代模式同构 mol2 ]\n\n"
        "关于嫁接 (graft, 不合法)\n"
        "   嫁接 = 切除模型原有取代基 + 接目标基团 (N(CCO)C / 绕丹宁等),\n"
        "   等于人工拼贴目标答案。原版本靠嫁接获得的 Tc=1.0 已删除, 不计入复现。\n"
        "   脚本与产物归档至 casestudy/8hzk/_deprecated_graft/。\n\n"
        "关键观察\n"
        "   · mol4 无 S, 二苯乙烯桥可由模型生成 → 生成导向, 靠迭代逼近\n"
        "   · mol2 含 S, 苯并噻吩模型不生成但有等电子体 → 编辑导向\n"
        "   · 两条链都止于骨架级; 进一步精确化属人工分子编辑"
    )
    fig.text(0.08, 0.86, overview, fontsize=10.0, va="top")

    fig.text(0.08, 0.34, "操作对比 (仅合法部分)", fontsize=13, fontweight="bold")

    table = [
        ("操作类型",   "mol4 路径",                          "mol2 路径"),
        ("起点",       "两来源汇聚 (CCN对接 + 生成stilbene)", "单一起点 (生成吲哚分子)"),
        ("骨架获取",   "sample_by_seed + 切割装饰",           "元素替换 NH→S (直接)"),
        ("迭代",       "固定骨架迭代 3-5 轮",                 "无 (一步替换到位)"),
        ("合法终点",   "Tc=0.58 (Murcko==mol4)",             "苯并噻吩骨架 (同构 mol2)"),
        ("供体来源",   "模型生成 (芳胺 NH2, 保留)",           "模型生成 (异丙烯氨基, 保留)"),
        ("链长度",     "3 步",                                "1 步"),
    ]
    y0 = 0.30
    row_h = 0.042
    for i, row in enumerate(table):
        y = y0 - i * row_h
        weight = "bold" if i == 0 else "normal"
        size = 10 if i == 0 else 9
        fig.text(0.08, y, row[0], fontsize=size, fontweight=weight)
        fig.text(0.24, y, row[1], fontsize=size, fontweight=weight)
        fig.text(0.62, y, row[2], fontsize=size, fontweight=weight)
        if i == 0:
            fig.add_artist(plt.Line2D(
                [0.08, 0.96], [y - 0.014, y - 0.014],
                color="black", lw=0.8))
            fig.add_artist(plt.Line2D(
                [0.08, 0.96], [y0 - len(table) * row_h + 0.028,
                               y0 - len(table) * row_h + 0.028],
                color="black", lw=0.5))

    fig.text(0.5, 0.025,
             "数据来源: MOLECULAR_PROVENANCE.md  ·  RDKit 2D 渲染  ·  "
             "matplotlib PdfPages",
             ha="center", fontsize=8, color="gray")

    pdf.savefig(fig)
    plt.close(fig)


def draw_merge_page(pdf, n1, n2, n3):
    """mol4 part 1: nodes 1 & 2 side-by-side → merge → node 3."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("mol4 分子级复现链  ·  Part 1/2: 起点汇聚",
                 fontsize=15, fontweight="bold", y=0.96)
    fig.text(0.5, 0.925,
             "两来源 (FluoDB CCN 受体 + 模型生成 stilbene) "
             "→ sample_by_seed 汇聚 → 含骨架产物",
             ha="center", fontsize=9.5, color="gray")
    ax_ov = draw_overlay(fig)

    # ---- top row: nodes 1, 2 side by side ----
    h_img_top = 0.22
    y_img_top = 0.66
    w_img_top = 0.42
    x1 = 0.04
    x2 = 0.54
    for x_img, node in [(x1, n1), (x2, n2)]:
        ax_img = fig.add_axes([x_img, y_img_top, w_img_top, h_img_top])
        ax_img.imshow(mol_to_image(node["smi"]))
        ax_img.axis("off")
        node_title(fig, node, x_img + w_img_top / 2,
                   y_img_top + h_img_top + 0.005, va="bottom")
        cap = (f'{node["role"]}\n'
               f'SMILES: {canon(node["smi"])}')
        fig.text(x_img, y_img_top - 0.005, cap,
                 fontsize=7.5, va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.4",
                           fc="#f7f7f7", ec="#cccccc"))

    # ---- merge arrows from nodes 1, 2 → node 3 ----
    y_a_top = 0.585
    y_a_bot = 0.525
    x_center = 0.50
    ax_ov.add_patch(FancyArrowPatch(
        (x1 + w_img_top / 2, y_a_top), (x_center - 0.04, y_a_bot),
        arrowstyle="-|>", mutation_scale=20, color="#666666", lw=1.8,
        connectionstyle="arc3,rad=-0.15"))
    ax_ov.add_patch(FancyArrowPatch(
        (x2 + w_img_top / 2, y_a_top), (x_center + 0.04, y_a_bot),
        arrowstyle="-|>", mutation_scale=20, color="#666666", lw=1.8,
        connectionstyle="arc3,rad=0.15"))
    ax_ov.text(x_center, (y_a_top + y_a_bot) / 2 + 0.010,
               n3["op_in"],
               fontsize=8.5, ha="center", va="center", color="#222222",
               bbox=dict(boxstyle="round,pad=0.35",
                         fc="#fff7d6", ec="#e0c060"))

    # ---- bottom: node 3 (full layout) ----
    y_img_bot = 0.08
    h_img_bot = 0.42
    w_img_bot = 0.46
    x_img_bot = 0.04
    ax_img3 = fig.add_axes([x_img_bot, y_img_bot, w_img_bot, h_img_bot])
    ax_img3.imshow(mol_to_image(n3["smi"], highlight=n3.get("final", False)))
    ax_img3.axis("off")
    node_title(fig, n3, x_img_bot + w_img_bot / 2,
               y_img_bot + h_img_bot + 0.005, va="bottom")
    node_text_panel(fig, n3, 0.54, y_img_bot + h_img_bot, va="top")

    # continuation hint
    fig.text(0.96, 0.025,
             "→ 续下页 (节点 4-5: 切割 → 固定骨架迭代 → 合法终点)",
             ha="right", fontsize=8.5, color="gray")

    pdf.savefig(fig)
    plt.close(fig)


def draw_chain_page(pdf, title, subtitle, nodes, footer=None):
    """Draw 1-3 nodes vertically with arrows between (op_in label).

    If `footer` is given, render it at the bottom of the page (legal-boundary
    statement shown beneath the chain's final node).
    """
    n = len(nodes)
    assert 1 <= n <= 3
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.96)
    if subtitle:
        fig.text(0.5, 0.925, subtitle, ha="center",
                 fontsize=9.5, color="gray")
    ax_ov = draw_overlay(fig)

    top_y = 0.88
    bot_y = 0.13 if footer else 0.04
    arrow_gap = 0.052
    available = top_y - bot_y - arrow_gap * (n - 1)
    slot_h = available / n
    h_img = slot_h - 0.025
    w_img = 0.46
    x_img = 0.04
    x_txt = 0.54

    for i, node in enumerate(nodes):
        y_top = top_y - i * (slot_h + arrow_gap)
        y_img = y_top - slot_h + 0.005
        ax_img = fig.add_axes([x_img, y_img, w_img, h_img])
        ax_img.imshow(mol_to_image(node["smi"],
                                   highlight=node.get("final", False)))
        ax_img.axis("off")
        node_title(fig, node, x_img + w_img / 2, y_top - 0.005, va="top")
        node_text_panel(fig, node, x_txt, y_top - 0.005, va="top")
        if i < n - 1:
            nxt = nodes[i + 1]
            y_a_top = y_img - 0.005
            y_a_bot = y_a_top - arrow_gap + 0.012
            x_a = x_img + w_img / 2
            draw_arrow(ax_ov, x_a, y_a_top, y_a_bot, nxt.get("op_in", ""))

    if footer:
        draw_chain_footer(fig, footer, y=0.035)

    pdf.savefig(fig)
    plt.close(fig)


# ----------------------------------------------------------------main
def main():
    print("Validating SMILES ...")
    all_nodes = MOL4_CHAIN + MOL2_CHAIN
    for node in all_nodes:
        m = Chem.MolFromSmiles(node["smi"])
        if m is None:
            raise SystemExit(f"FAIL parse node{node['idx']}: {node['smi']}")
        node["_canon"] = Chem.MolToSmiles(m)
        print(f"  node{node['idx']:>2}  heavy={m.GetNumHeavyAtoms():>3}  "
              f"final={node.get('final')}  {node['_canon'][:70]}")

    # verify final legal-endpoint nodes match EXPECTED (not target mol).
    # mol4 legal endpoint = iterative product (Tc=0.58, NOT mol4 itself).
    mol4_endpoint = Chem.MolToSmiles(
        Chem.MolFromSmiles("N#CC(=Cc1cccc(N)c1)c1ccc(-c2ccccc2)cc1"))
    assert MOL4_CHAIN[-1]["_canon"] == mol4_endpoint, (
        f"mol4 legal-endpoint mismatch:\n"
        f"  chain: {MOL4_CHAIN[-1]['_canon']}\n"
        f"  ref:   {mol4_endpoint}")
    # mol2 legal endpoint = element-substituted benzothiophene scaffold
    # (NOT mol2 itself; mol2 requires grafting rhodanine + N(CCO)C).
    mol2_endpoint = Chem.MolToSmiles(Chem.MolFromSmiles(
        "C=C(C)Nc1ccc(-c2cc3ccc(N4CCC[C@@H](O)C4)cc3s2)cc1"))
    assert MOL2_CHAIN[-1]["_canon"] == mol2_endpoint, (
        f"mol2 legal-endpoint mismatch:\n"
        f"  chain: {MOL2_CHAIN[-1]['_canon']}\n"
        f"  ref:   {mol2_endpoint}")
    print("OK: final nodes are LEGAL endpoints (iterative product / "
          "element-substituted scaffold).")
    print("   (target mols mol4/mol2 themselves require grafting and are "
          "NOT in the chain.)")

    # dry-run render every molecule to catch draw errors early
    print("\nDry-run rendering each molecule ...")
    for node in all_nodes:
        _ = mol_to_image(node["smi"], highlight=node.get("final", False))
        print(f"  node{node['idx']:>2}  rendered OK")

    print(f"\nWriting PDF: {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        draw_cover(pdf);                                  print("  page 1: cover")
        draw_merge_page(pdf, MOL4_CHAIN[0], MOL4_CHAIN[1],
                        MOL4_CHAIN[2]);                   print("  page 2: mol4 part 1 (merge n1+n2 → n3)")
        draw_chain_page(
            pdf,
            title="mol4 分子级复现链  ·  Part 2/2: 切割 → 迭代 → 合法终点",
            subtitle="切割装饰提取骨架 → 固定骨架迭代 "
                     "(Murcko==mol4, Tc=0.58)  [ == 骨架级复现 ]",
            nodes=MOL4_CHAIN[3:5],
            footer=MOL4_FOOTER,
        );                                                print("  page 3: mol4 part 2 (nodes 4, 5) + legal footer")
        draw_chain_page(
            pdf,
            title="mol2 分子级复现链: 元素替换 → 合法终点",
            subtitle="dummy_ligand #5133 (吲哚) → 元素替换 NH→S "
                     "获取苯并噻吩骨架 (与 mol2 同构)  [ == 骨架级复现 ]",
            nodes=MOL2_CHAIN[0:2],
            footer=MOL2_FOOTER,
        );                                                print("  page 4: mol2 (nodes 1, 2) + legal footer")

        meta = pdf.infodict()
        meta["Title"] = "mol4 & mol2 分子级复现前驱链 (骨架级, 合法)"
        meta["Subject"] = ("Molecular-level reproduction provenance "
                           "(8HZK case, scaffold-level legal endpoints)")

    size = os.path.getsize(OUT_PDF)
    print(f"\n[OK] PDF written: {OUT_PDF}")
    print(f"   Pages: 4  (cover + mol4x2 + mol2x1)")
    print(f"   Size : {size/1024:.1f} KB")
    print(f"   All {len(all_nodes)} molecule 2D structures rendered successfully.")
    print(f"   mol4 chain stops at node 5 (iterative product Tc=0.58).")
    print(f"   mol2 chain stops at node 2 (element-substituted scaffold).")


if __name__ == "__main__":
    main()
