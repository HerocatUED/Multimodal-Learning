# Multimodal-Learning
Project of Multimodal Learning (PKU 2023 Autumn)

This project is based on [Grounded-Diffusion](https://github.com/Lipurple/Grounded-Diffusion), 
but we modified the whole code base because the original codes are too ugly and hard to use.
We only remained the idea and reconsrtucted the whole project based on official stable diffusion code base.

TODO List：
- [x] Reproduce
- [x] Clean the code base
- [x] Try up-to-date stable diffusion models(we modified the whole code base)
- [x] Explorey frozen word embeddings
- [ ] Modify fusion module with advanced techniques
- [ ] Optional: Try segment a given image rather than segment generated images. 

## Requirements
1. Install [pytorch](https://pytorch.org/) (we use 2.1.1 with cuda 11.8)
2. Install requirements, see instructions under requirements folder
3. Make sure you have access to hugging face (If not, just put ```HF_ENDPOINT=https://hf-mirror.com``` before all commands bellow)

## Model Zoo
Put these models under `checkpoint` folder
https://drive.google.com/drive/folders/1HlagN6jVhmC_UbrOAy133LkN4Qgf2Scv?usp=sharing

## Train & Inference
Before training, please download the [checkpoint](https://drive.google.com/file/d/1JbJ7tWB15DzCB9pfLKnUHglckumOdUio/view) of the off-the-shelf detector into a folder called `checkpoint/`. 

See *command.txt*
	
## Acknowledgements
Many thanks to the code bases from [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [CLIP](https://github.com/openai/CLIP), [taming-transformers](https://github.com/CompVis/taming-transformers), [mmdetection](https://github.com/open-mmlab/mmdetection)
, [Stablility-AI:generative-models](https://github.com/Stability-AI/generative-models)