#!bin bash
git checkout train.py
git pull
sed -i 's/centos/raoby/g' train.py
