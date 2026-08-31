# Reproduction-route figures (SVG)

## sp1 flavone chain
| figure | role |
|---|---|
| sp1_target_m1 / m2 / m3 | targets (m1 159x, m2 16x reproduced) |
| sp1_skeleton | 21-atom all-C skeleton (6+6 fused core + six-membered ring) |
| sp1_seed_s1..s4 | inpainting seeds (intact rings; orange = retained O) |
| sp1_hit_m1 / sp1_hit_m2 | representative exact generated hits |

## T2 two-fragment chain
| figure | role |
|---|---|
| t2_source_1271 / t2_source_473 | two independent same-pocket (BER102) cold-start molecules; benzimidazole cores highlighted |
| t2_xfrag_seed | two-fragment anchor seed (native conformations, clash-free assembly); red = probed ring-N sites |
| t2_dimer_NN | model product: all-N isostere, green = model-generated CH2 bridge (568/915 poses) |
| t2_target_S | target T2 (S version) |
| t2_dimer_OO | element-scan variant (benzoxazole) |

Dual scores (phaseG): NN 0.674 (4/6) / -34.65; SS(=target) 0.692 (4/6) / -41.53;
OO 0.520 (3/6) / -32.93; berberine (positive reference) 0.595 (4/6) / -44.995.

Highlights: cyan/blue = the two ring systems; red = element-conversion sites
analysed; green = model-generated bridge; orange = hydroxyl/retained O.
Authoritative molecule files (single source of truth; no SDF duplicates
are kept in this directory):
- sp1: casestudy/sp1/{skeleton_flavone_sp1.sdf, inpaint_seeds/, gen_runA/valid_mols.sdf}
- T2: casestudy/kras_g4/t2_denovo/{fragment_sources/, xfrag_seeds/, gen_runX1/, phaseG_elem_scan/}
