#! /bin/sh
DATE=$(date '+%Y-%m-%d-%s')
python train.py > log.$DATE 2>&1 &