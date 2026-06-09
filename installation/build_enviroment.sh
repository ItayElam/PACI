#!/bin/bash

conda env create --file ./enviroment.yml 

LINE='export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/miniconda3/envs/PACI/lib/"'
FILE="$HOME/.bashrc"

grep -Fxq "$LINE" "$FILE" || echo "$LINE" >> "$FILE"
 