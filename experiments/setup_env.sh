#!/bin/bash

source ~/.bashrc
cd $WRKDIR/pbnn

python -m venv venv
source ./venv/bins/activate

pip install --upgrade pip
pip install -r requirements.txt
python setup.py develop
