#!bin bash
git checkout train.py download.py preprocess.py
git pull
sed -i 's/centos/raoby/g' train.py download.py preprocess.py
