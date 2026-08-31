# Reproduction-route figures (SVG)

## sp1 m1-route chain
| figure | role |
|---|---|
| sp1_target_m1 / m2 / m3 | targets (all three reached exactly by mutation) |
| sp1_source_core_499 | gen[499] chromanone core, cold-start original coordinates (11 atoms) |
| sp1_source_bring_12 | gen[12] phenol B-ring, cold-start original coordinates (7 atoms) |
| sp1_route_seed | assembled seed (499 core + 12 B-ring, clash-free) |
| sp1_mut_seed_sat / dehyd / dehyd_3OH / m1 / m2 / m3 | mutation ladder; mut_m1 == m1, mut_m2 == m2, mut_m3 == m3 |

Dual scores along the ladder: seed 0.691/-40.48 → dehyd 0.652/-39.82 → 3OH 0.652/-42.21
→ m1 0.652/-42.47 → m2 0.652/-44.74 → m3 0.652/-45.03 (rDock monotone, m3 best).

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
