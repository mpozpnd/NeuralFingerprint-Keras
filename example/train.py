#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
#
# Author:   Graphium
# URL:      http://tehutehu.com
# License:  MIT License
# Created:  2016-11-25
#
import os
import sys
sys.path.append(os.curdir)

from keras.layers import Dense, Activation
from keras import regularizers
from keras.models import Sequential

from nfp_keras.nfplayer import NFPLayer

import numpy as np
import argparse

ps = argparse.ArgumentParser()
ps.add_argument('input_data_x', type=str)
ps.add_argument('input_data_y', type=str)

args = ps.parse_args()

nb_epoch = 100
batch_size = 2
fp_len = 100

model = Sequential()
model.add(NFPLayer(fp_len, 62, 5, batch_input_shape=(batch_size, 24586,)))
model.add(Dense(1024, input_dim=fp_len))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

data_x = np.load(args.input_data_x)
data_y = np.load(args.input_data_y)

n_tr = 800
n_va = 100
n_te = 200

train_x = data_x[:n_tr]
valid_x = data_x[n_tr: n_tr + n_va]
test_x = data_x[n_tr + n_va: n_tr + n_va + n_te]

train_y = data_y[:n_tr]
valid_y = data_y[n_tr: n_tr + n_va]
test_y = data_y[n_tr + n_va: n_tr + n_va + n_te]

model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(valid_x, valid_y))

print(model.evaluate(test_x, test_y, batch_size=batch_size))

