#!/bin/bash

if [ ! -f delaney.csv ]; then
    wget -O delaney.csv https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-05-24-delaney/delaney-processed.csv 
fi

#python ./example/data_prep.py ./delaney.csv ./data
python ./example/train.py ./data_x.npy ./data_y.npy
#python ./example/compare2ecfp.py ./data_x.npy ./data_ecfp4.npy
