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
| t2_positive_ref_berberine | co-crystallized positive reference of the 7X8M BER102 site (RNAsmol 0.595 / rDock -44.995) |

Dual scores (phaseG): NN 0.674 (4/6) / -37.71; SS(=target) 0.692 (4/6) / -36.42;
OO 0.520 (3/6) / -28.13.

Highlights: cyan/blue = the two ring systems; red = element-conversion sites
analysed; green = model-generated bridge; orange = hydroxyl/retained O.
Each molecule also exported as a single-molecule SDF.
