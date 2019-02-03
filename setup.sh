#!/usr/bin/env bash

python --version >/dev/null 2>&1 || { echo >&2 "I require Python but it's not installed. Aborting."; exit 1; }
pip --version >/dev/null 2>&1 || { echo >&2 "I require PyPI but it's not installed. Aborting."; exit 1; }

clear

pip install -r requirements.txt

echo "done!"