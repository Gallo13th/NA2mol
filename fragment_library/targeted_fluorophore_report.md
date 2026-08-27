# Targeted Fluorophore Library — Build Report

- source_csv: `fluor_util_2/FluoDB-Full.csv`
- total_fluordb: 55168
- subset_size (D-pi-A): 7974
- build_time: 2026-07-24 16:04:01
- role_rules: priority: acceptor > donor > bridge; acceptor: C=S/C#N/nitro/imide/cyanoacrylic/conj-carbonyl; donor: tert-amine/aryl-sec-amine/TPA/carbazole/indole/phenothiazine/...; bridge: aromatic or C=C/C#C. Components = BRICS fragments + FluoDB-mined pharmacophore signatures.

## Bucket sizes
- **donors**: 226
- **acceptors**: 834
- **bridges**: 793
- total components: 1853
- 3D conformer success: 1851/1853

## 3D completeness (freq>=5, molblock non-empty rate)
- donors: 100.0% (71 components)
- acceptors: 100.0% (203 components)
- bridges: 100.0% (211 components)

## Top-10 per bucket (smiles | freq | n_ref)

### donors
- `CN(C)Cc1ccccc1` | freq=7465 | n_ref=7465 | smarts=`[NX3;H0]([#6])([#6])[#6]`
- `c1ccc(N(c2ccccc2)c2ccccc2)cc1` | freq=2498 | n_ref=2498 | smarts=`c1ccc(N(c2ccccc2)c2ccccc2)cc1`
- `c1ccccc1NC` | freq=517 | n_ref=517 | smarts=`[c][NX3;H1][#6]`
- `N1c2ccccc2Sc2ccccc21` | freq=493 | n_ref=493 | smarts=`N1c2ccccc2Sc2ccccc21`
- `c1ccc2c(c1)Nc1ccccc1S2` | freq=245 | n_ref=218 | smarts=`[#6]1:[#6]:[#6]:[#6]2:[#6](:[#6]:1)-[#7]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#16]-2`
- `Cc1ccc2c(c1)Sc1ccccc1N2` | freq=221 | n_ref=194 | smarts=`[#6]-[#6]1:[#6]:[#6]:[#6]2:[#6](:[#6]:1)-[#16]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#7]-2`
- `c1ccc2c(c1)NC1CCCC21` | freq=148 | n_ref=147 | smarts=`[#6]1:[#6]:[#6]:[#6]2:[#6](:[#6]:1)-[#7]-[#6]1-[#6]-[#6]-[#6]-[#6]-2-1`
- `c1ccc2nc3ccccc3cc2c1` | freq=128 | n_ref=128 | smarts=`[#6]1:[#6]:[#6]:[#6]2:[#7]:[#6]3:[#6]:[#6]:[#6]:[#6]:[#6]:3:[#6]:[#6]:2:[#6]:1`
- `c1ccc2[nH]ccc2c1` | freq=123 | n_ref=123 | smarts=`[#6]1:[#6]:[#6]:[#6]2:[#7H]:[#6]:[#6]:[#6]:2:[#6]:1`
- `CC1(C)CNc2ccccc21` | freq=112 | n_ref=111 | smarts=`[#6]-[#6]1(-[#6])-[#6]-[#7]-[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2-1`

### acceptors
- `O=CC=Cc1ccccc1` | freq=5141 | n_ref=5141 | smarts=`[#6]=[#6]-[#6]=[#8]`
- `N#Cc1ccccc1` | freq=4239 | n_ref=4239 | smarts=`[#7]#[#6]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1`
- `C=C(C#N)C(=O)O` | freq=3288 | n_ref=3288 | smarts=`[#6]=[#6](-[#6]#[#7])-[#6](=[#8])-[#8]`
- `N#CCC(=O)O` | freq=2737 | n_ref=2503 | smarts=`[#7]#[#6]-[#6]-[#6](=[#8])-[#8]`
- `O=[N+]([O-])c1ccccc1` | freq=695 | n_ref=695 | smarts=`[#8]=[#7+](-[#8-])-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1`
- `O=C1NC(=O)CC1` | freq=671 | n_ref=671 | smarts=`[#6](=O)-[#7]-[#6](=O)`
- `CC(=S)N` | freq=620 | n_ref=620 | smarts=`[#6]=[#16]`
- `N#CCC#N` | freq=425 | n_ref=371 | smarts=`[#7]#[#6]-[#6]-[#6]#[#7]`
- `O=C1CSC(=S)N1` | freq=306 | n_ref=272 | smarts=`[#8]=[#6]1-[#6]-[#16]-[#6](=[#16])-[#7]-1`
- `O=C1NC(=O)c2cccc3cccc1c23` | freq=200 | n_ref=178 | smarts=`[#8]=[#6]1-[#7]-[#6](=[#8])-[#6]2:[#6]:[#6]:[#6]:[#6]3:[#6]:[#6]:[#6]:[#6]-1:[#6]:2:3`

### bridges
- `c1ccccc1` | freq=9030 | n_ref=7781 | smarts=`[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1`
- `Cc1ccccc1` | freq=2561 | n_ref=2010 | smarts=`[#6]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1`
- `c1ccsc1` | freq=2277 | n_ref=2277 | smarts=`[#6]1:[#6]:[#6]:[#16]:[#6]:1`
- `C=CC=C` | freq=1793 | n_ref=1793 | smarts=`C=CC=C`
- `Cc1cccs1` | freq=1275 | n_ref=1112 | smarts=`[#6]-[#6]1:[#6]:[#6]:[#6]:[#16]:1`
- `c1cscn1` | freq=197 | n_ref=186 | smarts=`[#6]1:[#6]:[#16]:[#6]:[#7]:1`
- `CC1(C)c2ccccc2-c2ccccc21` | freq=176 | n_ref=99 | smarts=`[#6]-[#6]1(-[#6])-[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2-[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2-1`
- `c1ccc(Cc2ccccc2)cc1` | freq=145 | n_ref=127 | smarts=`[#6]1:[#6]:[#6]:[#6](-[#6]-[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2):[#6]:[#6]:1`
- `Cc1ccc(C)s1` | freq=145 | n_ref=134 | smarts=`[#6]-[#6]1:[#6]:[#6]:[#6](-[#6]):[#16]:1`
- `c1ccc2nsnc2c1` | freq=145 | n_ref=143 | smarts=`[#6]1:[#6]:[#6]:[#6]2:[#7]:[#16]:[#7]:[#6]:2:[#6]:1`
