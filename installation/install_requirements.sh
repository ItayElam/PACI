#!/bin/bash

pip3 install -r requirements.txt
pip3 install torchvision==0.16 --no-deps
pip3 install modified_torch/torch-2.4.0-cp39-cp39-linux_x86_64.whl
pip3 install transformers -U
