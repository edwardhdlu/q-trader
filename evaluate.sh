#!/usr/bin/env bash

python --version >/dev/null 2>&1 || { echo >&2 "I require Python but it's not installed. Aborting."; exit 1; }
pip --version >/dev/null 2>&1 || { echo >&2 "I require PyPI but it's not installed. Aborting."; exit 1; }

read -p "Please type model file (placed at ./models/model_epX), followed by [ENTER]: " MODEL_FILENAME
read -p "Please type filename.csv (without .csv, placed at ./data/*.csv), followed by [ENTER]: " DATA_FILENAME

clear

echo "Evaluate Q-trader model file $MODEL_FILENAME with data from file: $DATA_FILENAME"

python src/evaluate.py $DATA_FILENAME $MODEL_FILENAME

echo "done!"