"""
荧光分子功能片段库 (Fluorescent Molecule Functional Fragment Library)

该模块提供了荧光分子功能片段的提取、管理和查询功能。

功能片段定义:
- 电子供体 (Donors): 低价态的氧、氮、硫、碳、以及含有低价态上述原子的杂环以及苯环
  * 典型模式: 苯环、含N/O/S的杂环(如吡啶、噻吩、呋喃等)、氨基、羟基、烷氧基等

- 电子受体 (Acceptors): 含有高价态的氧(三价及以上)、氮(四价及以上)、磷(四价及以上)、硫(三价及以上)、卤素
  * 典型模式: 硝基、氰基、羰基、砜、卤素取代基、BODIPY核心等

- 连接子 (Linkers): 碳原子之间以单键、双键以及单双键交替的碳链
  * 典型模式: 乙烯基(-C=C-)、乙炔基(-C≡C-)、苯基桥连等
"""

import pickle
from typing import List, Dict, Set, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem


class FragmentLibrary:
    """
    荧光分子功能片段库类

    提供功能片段的加载、查询和分子片段识别功能。
    """

    # 预定义的片段SMARTS模式
    DONOR_PATTERNS = {
        # 苯环和芳香环
        'benzene': 'c1ccccc1',
        'phenyl': '[c]1[c][c][c][c][c]1',

        # 含氮杂环
        'pyridine': 'c1ccncc1',
        'pyrrole': 'c1cc[nH]c1',
        'imidazole': 'c1c[nH]cn1',
        'pyrazole': 'c1cn[nH]c1',
        'triazole': 'c1n[nH]cn1',
        'indole': 'c1ccc2[nH]ccc2c1',
        'carbazole': 'c1ccc2c(c1)[nH]c1ccccc12',
        'quinoline': 'c1ccc2ncccc2c1',

        # 含氧杂环
        'furan': 'c1ccoc1',
        'benzofuran': 'c1ccc2occc2c1',
        'chromene': 'c1ccc2occcc2c1',

        # 含硫杂环
        'thiophene': 'c1ccsc1',
        'benzothiophene': 'c1ccc2sccc2c1',
        'thiazole': 'c1cscn1',

        # 氨基和胺类
        'amino': '[NH2]',
        'primary_amine': '[NX3;H2]',
        'secondary_amine': '[NX3;H1]',
        'tertiary_amine': '[NX3;H0]',
        'aniline': 'c1ccccc1[NH2]',
        'dimethylamino': 'N(C)C',
        'diethylamino': 'N(CC)CC',

        # 羟基和烷氧基
        'hydroxyl': '[OH]',
        'methoxy': 'OC',
        'ethoxy': 'OCC',
        'phenoxy': 'Oc1ccccc1',

        # 醚类
        'ether': '[OX2]',

        # 硫醇和硫醚
        'thiol': '[SH]',
        'thioether': '[SX2]',

        # 三苯胺核心
        'triphenylamine': 'N(c1ccccc1)(c1ccccc1)c1ccccc1',

        # 给电子基团
        'methyl': 'C',
        'ethyl': 'CC',
        'isopropyl': 'C(C)C',
        'tert_butyl': 'C(C)(C)C',
    }

    ACCEPTOR_PATTERNS = {
        # 硝基
        'nitro': '[N+](=O)[O-]',
        'nitro_aromatic': '[N+](=O)[O-]c',

        # 氰基
        'cyano': 'C#N',

        # 羰基类
        'carbonyl': 'C=O',
        'aldehyde': '[CH]=O',
        'ketone': 'C(=O)C',
        'ester': 'C(=O)O',
        'carboxylic_acid': 'C(=O)[OH]',
        'amide': 'C(=O)N',

        # 砜和亚砜
        'sulfone': 'S(=O)(=O)',
        'sulfoxide': 'S(=O)',

        # 卤素
        'fluorine': '[F]',
        'chlorine': '[Cl]',
        'bromine': '[Br]',
        'iodine': '[I]',

        # 高价氮
        'quaternary_nitrogen': '[N+]',
        'pyridinium': '[n+]',

        # BODIPY核心
        'bodipy_core': '[B-]1(F)(F)[N+]2=CC=CC2=N1',

        # 硼酸酯
        'boronate': 'B(O)O',

        # 磷酸酯
        'phosphate': 'P(=O)(O)(O)',

        # 酰亚胺
        'imide': 'C(=O)NC(=O)',

        # 酸酐
        'anhydride': 'C(=O)OC(=O)',

        # 三氟甲基
        'trifluoromethyl': 'C(F)(F)F',
    }

    LINKER_PATTERNS = {
        # 乙烯基
        'vinylene': 'C=C',
        'trans_vinylene': 'C/C=C/C',
        'cis_vinylene': 'C/C=C\\C',

        # 乙炔基
        'ethynylene': 'C#C',

        # 苯基桥连
        'phenylene': 'c1ccc(cc1)',
        'meta_phenylene': 'c1cc(ccc1)',
        'para_phenylene': 'c1ccc(cc1)',

        # 共轭链
        'butadiene': 'C=CC=C',
        'hexatriene': 'C=CC=CC=C',

        # 杂原子桥连
        'amine_bridge': 'N',
        'ether_bridge': 'O',
        'thioether_bridge': 'S',

        # 羰基桥连
        'carbonyl_bridge': 'C(=O)',

        # 乙烯基苯
        'styryl': 'C=Cc1ccccc1',

        # 二苯乙烯
        'stilbene': 'c1ccccc1C=Cc1ccccc1',

        # 联苯
        'biphenyl': 'c1ccccc1-c2ccccc2',
    }

    def __init__(self, library_path: Optional[str] = None):
        """
        初始化片段库

        Args:
            library_path: 片段库pickle文件路径，如果为None则使用空库
        """
        self.donors: List[str] = []
        self.acceptors: List[str] = []
        self.linkers: List[str] = []
        self.stats: Dict = {}

        # 编译SMARTS模式
        self._donor_mols = self._compile_patterns(self.DONOR_PATTERNS)
        self._acceptor_mols = self._compile_patterns(self.ACCEPTOR_PATTERNS)
        self._linker_mols = self._compile_patterns(self.LINKER_PATTERNS)

        if library_path:
            self.load_library(library_path)

    def _compile_patterns(self, patterns_dict: Dict[str, str]) -> Dict[str, Chem.Mol]:
        """编译SMARTS模式为RDKit分子对象"""
        compiled = {}
        for name, smarts in patterns_dict.items():
            try:
                patt = Chem.MolFromSmarts(smarts)
                if patt is not None:
                    compiled[name] = patt
            except:
                pass
        return compiled

    def load_library(self, path: str) -> None:
        """
        从pickle文件加载片段库

        Args:
            path: pickle文件路径
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.donors = data.get('donors', [])
        self.acceptors = data.get('acceptors', [])
        self.linkers = data.get('linkers', [])
        self.stats = data.get('stats', {})

    def save_library(self, path: str) -> None:
        """
        保存片段库到pickle文件

        Args:
            path: pickle文件路径
        """
        data = {
            'donors': self.donors,
            'acceptors': self.acceptors,
            'linkers': self.linkers,
            'stats': self.stats,
            'donor_patterns': self.DONOR_PATTERNS,
            'acceptor_patterns': self.ACCEPTOR_PATTERNS,
            'linker_patterns': self.LINKER_PATTERNS,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def get_donors(self) -> List[str]:
        """获取供体片段列表"""
        return self.donors.copy()

    def get_acceptors(self) -> List[str]:
        """获取受体片段列表"""
        return self.acceptors.copy()

    def get_linkers(self) -> List[str]:
        """获取连接子片段列表"""
        return self.linkers.copy()

    def get_stats(self) -> Dict:
        """获取片段库统计信息"""
        return self.stats.copy()

    def find_fragments(self, smiles: str) -> Dict[str, List[str]]:
        """
        在分子中查找所有功能片段

        Args:
            smiles: 分子SMILES字符串

        Returns:
            包含供体、受体、连接子片段的字典
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'donors': [], 'acceptors': [], 'linkers': []}

        return {
            'donors': self._extract_fragments(mol, self._donor_mols),
            'acceptors': self._extract_fragments(mol, self._acceptor_mols),
            'linkers': self._extract_fragments(mol, self._linker_mols),
        }

    def _extract_fragments(self, mol: Chem.Mol, fragment_mols: Dict[str, Chem.Mol], 
                           min_atoms: int = 2) -> List[str]:
        """
        从分子中提取匹配的片段

        Args:
            mol: RDKit分子对象
            fragment_mols: 编译后的片段模式字典
            min_atoms: 最小原子数

        Returns:
            匹配的片段SMILES列表
        """
        matched_fragments = set()

        for name, patt in fragment_mols.items():
            try:
                matches = mol.GetSubstructMatches(patt, uniquify=True)
                for match in matches:
                    if len(match) >= min_atoms:
                        try:
                            frag_smiles = Chem.MolFragmentToSmiles(mol, list(match))
                            if frag_smiles and len(frag_smiles) > 0:
                                matched_fragments.add(frag_smiles)
                        except:
                            pass
            except:
                pass

        return sorted(list(matched_fragments))

    def has_donor(self, smiles: str) -> bool:
        """检查分子是否含有供体片段"""
        result = self.find_fragments(smiles)
        return len(result['donors']) > 0

    def has_acceptor(self, smiles: str) -> bool:
        """检查分子是否含有受体片段"""
        result = self.find_fragments(smiles)
        return len(result['acceptors']) > 0

    def has_linker(self, smiles: str) -> bool:
        """检查分子是否含有连接子片段"""
        result = self.find_fragments(smiles)
        return len(result['linkers']) > 0

    def get_fragment_count(self, smiles: str) -> Dict[str, int]:
        """
        获取分子中各类片段的数量

        Args:
            smiles: 分子SMILES字符串

        Returns:
            各类片段数量的字典
        """
        result = self.find_fragments(smiles)
        return {
            'donor_count': len(result['donors']),
            'acceptor_count': len(result['acceptors']),
            'linker_count': len(result['linkers']),
        }


# 便捷函数
def load_fragment_library(path: str = '/mnt/okcomputer/output/fluo_fragment_library.pkl') -> FragmentLibrary:
    """
    加载默认的荧光分子功能片段库

    Args:
        path: 片段库文件路径

    Returns:
        FragmentLibrary对象
    """
    return FragmentLibrary(path)


def extract_all_fragments(smiles: str, library: Optional[FragmentLibrary] = None) -> Dict[str, List[str]]:
    """
    从分子中提取所有功能片段

    Args:
        smiles: 分子SMILES字符串
        library: FragmentLibrary对象，如果为None则创建新实例

    Returns:
        包含供体、受体、连接子片段的字典
    """
    if library is None:
        library = FragmentLibrary()
    return library.find_fragments(smiles)


if __name__ == '__main__':
    # 测试代码
    lib = FragmentLibrary('/mnt/okcomputer/output/fluo_fragment_library.pkl')

    print("=" * 60)
    print("荧光分子功能片段库统计信息")
    print("=" * 60)
    stats = lib.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("供体片段示例 (前10个):")
    print("=" * 60)
    for frag in lib.get_donors()[:10]:
        print(f"  {frag}")

    print("\n" + "=" * 60)
    print("受体片段示例 (前10个):")
    print("=" * 60)
    for frag in lib.get_acceptors()[:10]:
        print(f"  {frag}")

    print("\n" + "=" * 60)
    print("连接子片段示例 (前10个):")
    print("=" * 60)
    for frag in lib.get_linkers()[:10]:
        print(f"  {frag}")

    # 测试分子
    test_smiles = "c1ccc(N(c2ccccc2)c2ccccc2)cc1"  # 三苯胺
    print("\n" + "=" * 60)
    print(f"测试分子: {test_smiles}")
    print("=" * 60)
    result = lib.find_fragments(test_smiles)
    print(f"供体片段: {result['donors']}")
    print(f"受体片段: {result['acceptors']}")
    print(f"连接子片段: {result['linkers']}")
