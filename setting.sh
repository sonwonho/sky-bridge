#!/bin/bash

apt update && apt upgrade -y
apt install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install python3.12 -y
apt install python3-distutils -y
apt install curl -y
apt install python3-setuptools -y
python3.12 -m easy_install install pip

python3.12 -m venv .venv
source .venv/bin/activate
pip3.12 install -r requirements.txt

apt install libglu1-mesa-dev -y
tar -xvf venv.tar
bash utils/standalone_embed.sh start
# python3.12 input_milvus.py