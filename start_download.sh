#! /bin/sh
DATE=$(date '+%Y-%m-%d-%s')
python download.py > log.download.$DATE 2>&1 &