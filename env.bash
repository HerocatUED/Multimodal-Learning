#! /bin/bash
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