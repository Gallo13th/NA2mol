# Process-molecule audit figures

Every molecule used at every stage of the reproduction routes, drawn as 2D
topology with key-fragment highlights; each also exported as a single-molecule
SDF next to its figure.

## T2 staged de-novo chain (casestudy/kras_g4/t2_denovo/)

| prefix | stage | molecules |
|---|---|---|
| t2_s1_lib_* | S1 cold-start library members (full structures) | gen2k1271 / gen2k473 / gen2k348 |
| t2_s1_core_* | S1 trimmed 5+6 cores (9 atoms) | one per library member; red = N->S site |
| t2_s2a_dimer* | S2a all-N isoelectronic dimers | 1271x1271, 1271x348, 1271x473 |
| t2_s2b_pose_gp* | S2b gated pocket poses | gp1/gp2/gp3 (pre-replacement) |
| t2_s2c_seed_* | S2c two-benzothiazole anchor seeds | gp1-3 endorsed + centroid0-3 |
| t2_s3_nh_* | S3 inpainting products | NH-bridged, 3 conformers (green = bridge N) |
| t2_s4_product | S4 final N->C product | canonical SMILES == target T2 |
| t2_target | reference | |

## sp1 flavone chain (casestudy/sp1/)

| prefix | stage | molecules |
|---|---|---|
| sp1_target_m1/m2/m3 | targets | m1 159x, m2 16x, m3 not reproduced |
| sp1_s0_skeleton | S0 21-atom all-C skeleton | 6+6 fused core + six-membered ring |
| sp1_s1_seed_s1..s4 | inpainting seeds | intact rings; orange = retained O |
| sp1_p_m1_hit* / sp1_p_m2_hit1 | exact generated hits | poses from gen_runA |
| sp1_p_variant1..3 | scaffold variants | same BM scaffold, model-chosen substituents |

## Highlight scheme

cyan = ring system 1 (fused 5+6 / 6+6 core); blue = ring system 2;
red = N->S conversion sites; green = bridge atom (N->C); orange = oxygen atoms
(hydroxyls / seed-retained); magenta = reserved.
