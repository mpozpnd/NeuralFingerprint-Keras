#!/bin/bash

if [ $# -ne 1 ] ; then
    echo 'Usage: ./test.sh [ dataprep | compare | trainNFP | trainECFP ]'
    exit 1
fi


if [ $1 = dataprep ] ; then
    if [ ! -f delaney.csv ]; then
        wget -O delaney.csv https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-05-24-delaney/delaney-processed.csv 
    fi
    python ./example/data_prep.py ./delaney.csv ./data

elif [ $1 = compare ] ; then
    python ./example/compare2ecfp.py ./data_x.npy ./data_ecfp4.npy

elif [ $1 = trainNFP ] ; then
    python ./example/train.py ./data_x.npy ./data_y.npy

elif [ $1 = trainECFP ] ; then
    python ./example/train_ecfp.py ./data_ecfp4.npy ./data_y.npy

else
    echo 'Usage: ./test.sh [ dataprep | compare | trainNFP | trainECFP ]'
    exit 1
fi
