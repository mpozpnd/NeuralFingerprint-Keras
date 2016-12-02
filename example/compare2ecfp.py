#!/usr/bin/env python
# -*- coding: utf-8 -*-

# vim: set fileencoding=utf-8 :
#
# Author:   Graphium
# URL:      http://tehutehu.com
# License:  MIT License
# Created:  2016-11-27
#
import os
import sys
sys.path.append(os.curdir)

import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

from nfp_keras.nfplayer import NFPLayer

import argparse

ps = argparse.ArgumentParser()
ps.add_argument('input_data', type=str)
ps.add_argument('ecfp4_data', type=str)
args = ps.parse_args()

def distance(x, y):
    tmp = np.vstack([x, y])
    return 1 - tmp.min(axis=0).sum() / tmp.max(axis=0).sum()

ecfp = np.load(args.ecfp4_data)[:50]
data_x = np.load(args.input_data)[:50]

layer = NFPLayer(2048, 62, 5, large_weights=True, batch_input_shape=(2, 24586,))
layer.build(input_shape=24586)
datas = [data_x[i: i+2] for i in range(0,len(data_x),2)]
nfp = np.vstack([layer.call(K.variable(x)).eval() for x in datas])

dists_ecfp = np.hstack([distance(x, y) for x in ecfp for y in ecfp])
dists_nfp = np.hstack([distance(x, y) for x in nfp for y in nfp])

print(dists_ecfp.sum())
print(dists_nfp.sum())
plt.plot(dists_ecfp, dists_nfp, '.')
plt.xlabel('distances using ECFP')
plt.ylabel('distances using NFP')
plt.savefig('compare.png')
print('correlation between  two FP:%f' % np.corrcoef(dists_ecfp, dists_nfp)[0, 1])



