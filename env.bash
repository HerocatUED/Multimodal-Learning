#! /bin/bash

conda create -n mml python==3.8.5
conda activate mml
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

cp -p coco_80_class.txt src/mmdetection/demo/

# other modules from Github
mkdir src
cd src

# CLIP
git clone https://github.com/openai/CLIP.git
cd CLIP
pip install -e .
cd ..

# mmdetection
mim install mmengine
mim install "mmcv>=2.0.0"
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -e .
cd ..

# taming transformer
git clone https://github.com/CompVis/taming-transformers
cd taming-transformers
pip install -e .
cd ..

# mmtrack
git clone https://github.com/open-mmlab/mmtracking.git
cd mmtracking
pip install -e .
cd ..