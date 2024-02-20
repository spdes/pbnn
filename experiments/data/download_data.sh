#!/bin/bash
#
# regression
#
# boston
wget https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data
# concrete
wget https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls
# energy
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx
# kin8mn
#wget http://mldata.org/repository/data/download/csv/uci-20070111-kin8nm
wget https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.arff -O dataset_2175_kin8nm.csv
# naval
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip
unzip -o UCI\ CBM\ Dataset.zip
# power
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip
unzip -o CCPP.zip
# protein
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv
# winered
wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
# winewhite
wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
# yacht
wget https://archive.ics.uci.edu/ml/machine-learning-databases//00243/yacht_hydrodynamics.data
# year prediction MSD
wget https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip
unzip -o yearpredictionmsd.zip

# classification
# australian credit approval
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/australian
# breast cancer Wisconsin
wget https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip
unzip -o breast+cancer+wisconsin+diagnostic.zip
# ionosphere
wget https://archive.ics.uci.edu/static/public/52/ionosphere.zip
unzip -o ionosphere.zip
# glass identification
wget https://archive.ics.uci.edu/static/public/42/glass+identification.zip
unzip -o glass+identification.zip
# vehicles silhouettes
#wget https://archive.ics.uci.edu/static/public/149/statlog+vehicle+silhouettes.zip
#unzip statlog+vehicle+silhouettes.zip
# waveform

# digits

# satellite
wget https://archive.ics.uci.edu/static/public/146/statlog+landsat+satellite.zip
unzip -o statlog+landsat+satellite.zip
