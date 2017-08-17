#! /bin/sh
DATE=$(date '+%Y-%m-%d-%s')
python preprocess.py > log.preprocess.$DATE 2>&1 &