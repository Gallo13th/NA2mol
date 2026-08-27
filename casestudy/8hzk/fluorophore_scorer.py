#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPEC_01 §6 — FluorophoreScorer

加载靶向荧光构件库, 对一个分子做 D-π-A 完整性打分。
匹配方式: 对每个构件用其 `smarts` 做 mol.GetSubstructMatch, 命中即计。
"""
import os
import pickle

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_LIB = os.path.join(REPO_ROOT, 'fragment_library', 'targeted_fluorophore.pkl')


class FluorophoreScorer:
    """对分子评估其 D-π-A (donor-pi-acceptor) 构件覆盖度。"""

    def __init__(self, library_path=DEFAULT_LIB):
        with open(library_path, 'rb') as f:
            self.library = pickle.load(f)
        self.donors = self.library['donors']
        self.acceptors = self.library['acceptors']
        self.bridges = self.library['bridges']
        # 预编译 SMARTS, 缓存
        self._pat = {}
        for comps in (self.donors, self.acceptors, self.bridges):
            for c in comps:
                self._pat[c['smiles']] = Chem.MolFromSmarts(c['smarts'])

    def _match(self, mol, comps):
        hits = []
        for c in comps:
            pat = self._pat.get(c['smiles'])
            if pat is None:
                pat = Chem.MolFromSmarts(c['smarts'])
                self._pat[c['smiles']] = pat
            if pat is not None and mol.HasSubstructMatch(pat):
                hits.append(c['smiles'])
        return hits

    def score_molecule(self, mol) -> dict:
        """
        返回:
          {
            'donor_hits': int, 'acceptor_hits': int, 'bridge_hits': int,
            'is_complete_dpa': bool,        # 三类均 >0
            'matched_donors': [smiles...],
            'matched_acceptors': [smiles...],
            'matched_bridges': [smiles...],
          }
        """
        if mol is None:
            return self._empty()
        d = self._match(mol, self.donors)
        a = self._match(mol, self.acceptors)
        b = self._match(mol, self.bridges)
        return {
            'donor_hits': len(d),
            'acceptor_hits': len(a),
            'bridge_hits': len(b),
            'is_complete_dpa': bool(d) and bool(a) and bool(b),
            'matched_donors': d,
            'matched_acceptors': a,
            'matched_bridges': b,
        }

    @staticmethod
    def _empty():
        return {
            'donor_hits': 0, 'acceptor_hits': 0, 'bridge_hits': 0,
            'is_complete_dpa': False,
            'matched_donors': [], 'matched_acceptors': [], 'matched_bridges': [],
        }


if __name__ == '__main__':
    s = FluorophoreScorer()
    import sys
    if len(sys.argv) > 1:
        m = Chem.MolFromSmiles(sys.argv[1])
        print(s.score_molecule(m))
