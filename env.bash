#! /bin/bash
conda create -n mml python==3.8.5
conda activate mml
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
makedir src
cd src
# CLIP
git clone git@https://github.com/openai/CLIP.git
cd CLIP
pip install -e .
cd ..
# mmtrack
git clone git@github.com:open-mmlab/mmtracking.git
cd mmtracking
pip install -e .
# mmdetection (maybe)