import sys
import os
sys.path.append(os.curdir)

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from nfp_keras.features import gen_feature_vector

import argparse
import pandas as pd


ps = argparse.ArgumentParser() 
ps.add_argument("data", type=str, help="csv file")
ps.add_argument("out_prefix", type=str)
args = ps.parse_args()

df = pd.read_csv(args.data)

smiles = df["smiles"]
data_raw = [gen_feature_vector(Chem.MolFromSmiles(x)) for x in smiles]
max_len = max([x.shape for x in data_raw])[0]
data = np.zeros([len(data_raw), max_len])

for i, row in enumerate(data_raw):
    data[i, :row.shape[0]] = row

data = np.vstack(data)
np.save('%s_x.npy' % args.out_prefix, data)
np.save('%s_y.npy' % args.out_prefix, df["measured log solubility in mols per litre"])

fp = []
for sm in smiles:
    fp.append(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(sm), 4))
np.save('%s_ecfp4.npy' % args.out_prefix, np.vstack(fp))

