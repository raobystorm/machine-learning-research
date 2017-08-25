#!bin bash
git checkout train.py download.py preprocess.py test_model.py
git pull
sed -i 's/centos/raoby/g' train.py download.py preprocess.py test_model.py
