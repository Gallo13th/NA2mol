import os
import rdkit
from rdkit import Chem
import glob
from rdkit import RDConfig
from rdkit import RDLogger
import sys, os
from Bio.Data import PDBData
RDLogger.DisableLog('rdApp.*')
RDLogger.logger().setLevel(RDLogger.ERROR)
def merge_sdfs(sdf_list, output_filename):
    """将所有成功的分子合并到一个 SDF 文件"""
    writer = Chem.SDWriter(output_filename)
    count = 0
    for sdf in sdf_list:
        suppl = Chem.SDMolSupplier(sdf)
        for mol in suppl:
            if mol:
                writer.write(mol)
                count += 1
    writer.close()
    print(f"✅ 已将 {count} 个分子合并保存至: {output_filename}")

import numpy as np
from rdkit.Chem import QED, Descriptors, rdMolDescriptors
from util import sascorer

import stat
import subprocess

def Minimize(input_sdf, pdb, workdir):
    smina = "./smina.static"
    if not os.access(smina, os.X_OK):
        os.chmod(smina, os.stat(smina).st_mode | stat.S_IXUSR)
    # tmpdir = os.path.join(workdir, 'tmp')
    output_sdf = os.path.join(workdir, input_sdf.split("/")[-1].split(".sdf")[0]+"_lig_score.sdf")
    command = f"{smina} -r {pdb} -l {input_sdf} -o {output_sdf} --minimize --minimize_iters 1000 --cpu 1" #  >> {workdir}/smina.log 2>&1"
    # print("Running minimization command:", command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    #print(result)
    return output_sdf

# ================= 新增：种子质量评分辅助函数 =================
def calc_seed_quality(frag_mol: Chem.Mol):
    """
    计算片段的综合质量得分 (Score)。
    Score = w1*LBE + w2*QED + w3*NCI_Efficiency - w4*Complexity
    """
    try:
        num_heavy = frag_mol.GetNumHeavyAtoms()
        if num_heavy == 0: return -100, {}, False
        affinity = float(frag_mol.GetProp("minimizedAffinity"))
        # 1. LBE (Ligand Binding Efficiency) 估算
        lbe = -affinity / num_heavy  # parent_affinity 通常为负，取反变正

        # 2. QED (Quantitative Estimate of Drug-likeness)
        frag_noH = Chem.RemoveHs(frag_mol)
        sascore = sascorer.calculateScore(frag_noH)

        qed_val = QED.qed(frag_mol)
        mw = Descriptors.HeavyAtomMolWt(frag_mol)
        Iteration = int(frag_mol.GetProp("Iteration"))
        
        metrics = {
            "LBE": lbe, "QED": qed_val, "SA Score": sascore, "Affinity": affinity, "MW":mw, "Iteration": Iteration
        }
        
        return metrics

    except Exception as e:
        print(e)
        return {}
from tqdm import tqdm
import pandas as pd

results = []
# output_dir = "/home/jovyan/lt/DiffEDMol-unitrans/output/20260209_dude_nofrag_genbyseed/"
# output_dir = "/home/jovyan/lt/DiffEDMol-unitrans/output/20260218_rna_nofrag_genbyseed/"
# for dude_cand in tqdm(os.listdir(output_dir)):
#     if '.' in dude_cand: continue
#     if not os.path.exists(f"{output_dir}{dude_cand}/{dude_cand}.sdf"):
#         continue
#     try:
#         suppl = Chem.SDMolSupplier(f"{output_dir}{dude_cand}/{dude_cand}.sdf")
#         for idx, mol in tqdm(enumerate(suppl)):
#             if mol is None: continue
#             results.append({'index':idx,**calc_seed_quality(mol),'pdb_id':dude_cand})
#     except:
#         continue
    
# df = pd.DataFrame(results)
# print(df.describe())
# df.to_csv('20260218_robin_original.csv')

df = pd.read_csv('20260218_robin_original.csv', index_col=0)
df['pdb_id'] = df['pdb_id'].apply(lambda x: x.split("_")[0])
allowed_pdbs = ['8d5o', '3f2q', '8d5l', '1yit', '8d2b', '6wzs', '2o3x', '6e1v',
       '2lwk', '6wzr', '5c45', '3irw', '6e1s', '2ogo', '8d2a', '8d28',
       '2g9c', '6e1u', '2ku0', '1yrj', '2ogn', '3bnq', '5xi1', '3f4g',
       '2ktz', '2kxm', '6dn2']
df = df[df['pdb_id'].isin(allowed_pdbs)]
print(df[(df['Affinity'] < 0) & (df['Iteration'] <= 10) & (df['MW'] > 360)].describe())
print(df[(df['Affinity'] < 0) & (df['Iteration'] > 10) & (df['MW'] > 360)].describe())
print(df[(df['Affinity'] < 0) & (df['SA Score'] < 5) & (df['QED'] > 0.5) & (df['Iteration'] <= 10) & (df['MW'] > 240)].describe())
print(df[(df['Affinity'] < 0) & (df['SA Score'] < 5) & (df['QED'] > 0.5) & (df['Iteration'] > 15) & (df['MW'] > 240)].describe())

# df = pd.read_csv('/home/jovyan/lt/MolCRAFT/dude_original.csv', index_col=0)

# print(df[(df['Affinity'] < 0) & (df['SA Score'] < 5) & (df['QED'] > 0.5) & (df['MW'] > 240)].describe())