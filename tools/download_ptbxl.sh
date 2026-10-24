#!/usr/bin/env bash
# Downloads PTB-XL v1.0.3 (CC BY 4.0) from PhysioNet into data/ptbxl.
# Wagner P. et al., PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3), PhysioNet (2022). https://doi.org/10.13026/kfzx-aw45
set -e
cd "$(dirname "$0")/.."
mkdir -p data/ptbxl
wget -r -N -c -np -nH --cut-dirs=3 -P data/ptbxl https://physionet.org/files/ptb-xl/1.0.3/
