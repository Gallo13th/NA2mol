import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, Crippen, QED, rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer
import posebusters
import posecheck

# ==========================================
# 1. 全局配置与初始化
# ==========================================

# 关闭 RDKit 日志
RDLogger.DisableLog('rdApp.*')

# 初始化指纹生成器 (ECFP4)
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# 初始化评估工具 (全局只初始化一次，避免循环内重复加载)
PC_INSTANCE = posecheck.PoseCheck()
PB_INSTANCE = posebusters.PoseBusters(config="dock")

# 定义模型配置列表
# 每个模型包含：名称、输入文件获取逻辑、输出文件名
MODEL_CONFIGS = [
    # {
    #     "name": "MolCRAFT",
    #     "output_csv": "molcraft_dude_results.csv",
    #     "get_sdf_files": lambda pdb_id: glob.glob(f"./output/molcraft_dude_minimized/{pdb_id}/*.sdf")
    # },
    # {
    #     "name": "MolCRAFT (Original)",
    #     "output_csv": "molcraft_orig_dude_results.csv",
    #     "get_sdf_files": lambda pdb_id: [f"./output/molcraft_orig_dude/{pdb_id}/merged_all_lig_score.sdf"]
    # },
    # {
    #     "name": "RNA2smol (Free)",
    #     "output_csv": "diffedmol_dude0303_results.csv",
    #     "get_sdf_files": lambda pdb_id: [f"output/20260303_dude_nofrag_genbyfree/dude_nh_{pdb_id}_noH/dude_nh_{pdb_id}_noH.sdf"]
    # },
    {
        "name": "Pocket2Mol",
        "output_csv": "Pocket2Mol_dude_results.csv",
        "get_sdf_files": lambda pdb_id: [f"output/pkt2mol/{pdb_id}_ligands_minimized.sdf"]
    },
    # {
    #     "name": "ed2mol",
    #     "output_csv": "ed2mol_dude_results.csv",
    #     "get_sdf_files": lambda pdb_id: [f"./output/ed2mol_dude/{pdb_id}.sdf"]
    # }
]

# 路径配置
PROTEIN_DIR = './DUDE_complex/DUDE_proteins/'
ACTIVE_CSV_PATH = "./DUDE_active_compounds.csv"

# ==========================================
# 2. 辅助函数
# ==========================================

def clean_pdb_ids(df):
    """修复 PDB ID 的科学计数法错误格式"""
    if 'pdb_id' not in df.columns:
        return df
    
    df = df.copy()
    df['pdb_id'] = df['pdb_id'].astype(str).str.strip().str.lower()
    
    # 手动修复特定错误
    replace_dict = {
        '3.00e+37': '3e37', '3e+37': '3e37', '3.0e+37': '3e37', '3e37.0': '3e37', '3.000000e+37': '3e37',
        '1.00e+66': '1e66', '1e+66': '1e66', '1.0e+66': '1e66', '1e66.0': '1e66', '1.000000e+66': '1e66'
    }
    df['pdb_id'] = df['pdb_id'].replace(replace_dict)
    
    # 正则修复通用格式
    df['pdb_id'] = df['pdb_id'].str.replace(r'^3[eE][\+\-]?\d+$', '3e37', regex=True)
    df['pdb_id'] = df['pdb_id'].str.replace(r'^1[eE][\+\-]?\d+$', '1e66', regex=True)
    
    return df

def get_mol_descriptors(mol, pocket_path):
    """计算分子描述符 (复用全局初始化的评估工具)"""
    descriptor_dict = {}
    try:
        # 基本性质
        descriptor_dict['mw'] = Descriptors.MolWt(mol)
        descriptor_dict['heavy_atoms'] = mol.GetNumHeavyAtoms()
        descriptor_dict['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
        descriptor_dict["SMILES"] = Chem.MolToSmiles(mol)
        
        # 类药性
        descriptor_dict['qed'] = QED.qed(mol)
        descriptor_dict['logp'] = Crippen.MolLogP(mol)
        descriptor_dict['tpsa'] = Descriptors.TPSA(mol)
        
        # 合成可及性
        try:
            descriptor_dict['sa_score'] = sascorer.calculateScore(mol)
        except:
            descriptor_dict['sa_score'] = 3.0
            
        # 从 SDF 属性中读取 (如果存在)
        for prop in ['minimizedAffinity', 'minimizedRMSD', 'GenerateMode']:
            try:
                val = mol.GetProp(prop)
                if prop == 'minimizedAffinity' or prop == 'minimizedRMSD':
                    descriptor_dict[prop.replace('minimized', '').lower()] = float(val)
                else:
                    descriptor_dict[prop] = val
            except:
                pass
        
        # PoseCheck (Strain Energy)
        try:
            PC_INSTANCE.load_ligands_from_mols([mol])
            strain_list = PC_INSTANCE.calculate_strain_energy()
            descriptor_dict['strain'] = strain_list[0]
        except Exception as e:
            # print(f"Error calculating strain energy: {e}")
            pass
        
    except Exception as e:
        print(f"Error calculating descriptors: {e}")
        
    return descriptor_dict

def load_active_data():
    """加载并预处理活性化合物数据"""
    if not os.path.exists(ACTIVE_CSV_PATH):
        raise FileNotFoundError(f"Active compounds file not found: {ACTIVE_CSV_PATH}")
    
    bsl_df = pd.read_csv(ACTIVE_CSV_PATH)
    bsl_df = clean_pdb_ids(bsl_df)
    return bsl_df

def get_active_fps_for_pdb(pdb_id, bsl_df):
    """为特定 PDB ID 获取活性分子的 SMILES 和指纹 (带缓存逻辑)"""
    active_smis = bsl_df[bsl_df['pdb_id'] == pdb_id]['canonical_smiles'].unique().tolist()
    
    if not active_smis:
        return [], []
    
    fps = []
    valid_smis = []
    for smi in active_smis:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                fps.append(MORGAN_GEN.GetFingerprint(mol))
                valid_smis.append(smi)
            except:
                continue
    return valid_smis, fps

# ==========================================
# 3. 核心处理逻辑
# ==========================================

def process_single_model(config, bsl_df, pockets):
    """处理单个模型的所有数据"""
    print(f"\n{'='*20} Processing Model: {config['name']} {'='*20}")
    all_res = []
    
    for pocket in tqdm(pockets, desc=f"Scanning Pockets ({config['name']})"):
        pdb_id = pocket[:4]
        pocket_pdb = os.path.join(PROTEIN_DIR, f'{pdb_id}_protein_wo_ligand.pdb')
        
        # 获取该模型的 SDF 文件列表
        sdf_files = config['get_sdf_files'](pdb_id)
        # 过滤掉不存在的文件
        sdf_files = [f for f in sdf_files if os.path.exists(f)]
        
        if not sdf_files:
            continue
            
        # 获取活性分子指纹
        active_smis, active_fps = get_active_fps_for_pdb(pdb_id, bsl_df)
        if not active_smis:
            print(f"No active compounds found for pdb_id={pdb_id}")
            continue

        results = []
        
        # 遍历所有生成的 SDF 文件
        for sdf_path in tqdm(sdf_files):
            try:
                supplier = Chem.SDMolSupplier(sdf_path, removeHs=True)
            except Exception as e:
                print(f"Error reading {sdf_path}: {e}")
                continue
                
            for idx, mol in enumerate(supplier):
                if mol is None:
                    continue
                
                # 计算描述符
                result = get_mol_descriptors(mol, pocket_pdb)
                
                # 计算相似度
                if active_fps:
                    try:
                        gen_fp = MORGAN_GEN.GetFingerprint(mol)
                        sims = DataStructs.BulkTanimotoSimilarity(gen_fp, active_fps)
                        max_sim = max(sims) if sims else 0.0
                        result['max_ecfp4_tanimoto'] = max_sim
                        # 获取最相似的活性分子 SMILES
                        result['active_smi'] = active_smis[sims.index(max_sim)]
                    except:
                        result['max_ecfp4_tanimoto'] = None
                        result['active_smi'] = None
                else:
                    result['max_ecfp4_tanimoto'] = None
                    result['active_smi'] = None

                result['mol_index'] = idx
                result['source_file'] = os.path.basename(sdf_path)
                results.append(result)
        
        if results:
            result_df = pd.DataFrame(results)
            result_df['pdb_id'] = pdb_id
            result_df['mode'] = config['name']
            all_res.append(result_df)
    
    if not all_res:
        print(f"No results generated for {config['name']}")
        return

    # 合并结果
    molcraft_df = pd.concat(all_res, ignore_index=True)
    
    # 后处理与过滤
    try:
        # 保留亲和力有效的分子
        all_df_dockok = molcraft_df[molcraft_df['affinity'] < 0]
    except KeyError:
        # 如果没有 affinity 列，保留全部
        all_df_dockok = molcraft_df.copy()
    
    # 计算 LBE
    if 'heavy_atoms' in all_df_dockok.columns and 'affinity' in all_df_dockok.columns:
        # 避免除以零
        all_df_dockok['LBE'] = all_df_dockok.apply(
            lambda x: x['affinity'] / x['heavy_atoms'] if x['heavy_atoms'] > 0 else np.nan, 
            axis=1
        )
    
    # 保存结果 (遵循原逻辑：保存 affinity < 0 的结果)
    output_file = config['output_csv']
    all_df_dockok.to_csv(output_file, index=False)
    print(f"Saved {len(all_df_dockok)} rows to {output_file}")

# ==========================================
# 4. 主程序入口
# ==========================================

def main():
    # 1. 加载活性数据 (只加载一次)
    print("Loading active compounds...")
    bsl_df = load_active_data()
    
    # 2. 获取蛋白质口袋列表
    if not os.path.exists(PROTEIN_DIR):
        raise FileNotFoundError(f"Protein directory not found: {PROTEIN_DIR}")
    pockets = os.listdir(PROTEIN_DIR)
    print(f"Found {len(pockets)} pockets in {PROTEIN_DIR}")
    
    # 3. 遍历配置处理每个模型
    for config in MODEL_CONFIGS:
        try:
            process_single_model(config, bsl_df, pockets)
        except Exception as e:
            print(f"Critical error processing {config['name']}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()