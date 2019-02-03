#!/usr/bin/env bash

python --version >/dev/null 2>&1 || { echo >&2 "I require Python but it's not installed. Aborting."; exit 1; }
pip --version >/dev/null 2>&1 || { echo >&2 "I require PyPI but it's not installed. Aborting."; exit 1; }


read -p "Please type filename.csv (without .csv, placed at ./data/*.csv), followed by [ENTER]:" DATA_FILENAME
read -p "Please type integer number of windows size, followed by [ENTER]:" WINDOW_SIZE
read -p "Please type integer number of episodes, followed by [ENTER]:" EPISODES_COUNT

clear

echo "Launch Q-trader with file: $DATA_FILENAME window_size=$WINDOW_SIZE episodes=$EPISODES_COUNT"

python src/train.py $DATA_FILENAME $WINDOW_SIZE $EPISODES_COUNT

echo "done!"