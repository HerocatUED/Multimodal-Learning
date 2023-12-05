# Multimodal-Learning
Project of Multimodal Learning (PKU 2023 Autumn)

This project is based on [Grounded-Diffusion](https://github.com/Lipurple/Grounded-Diffusion)

TODO List：
- [ ] Reproduce
- [ ] Try stable diffusion UI
- [ ] Try open-vocabulary pretrained word embedding rather than CLIP
- [ ] Modify fusion module with advanced techniques
- [ ] Optional: Try segment a given image rather than segment generated images. 

## Requirements
1. Install [pytorch](https://pytorch.org/) (we use 2.1.1 with cuda 11.8)
2. Run `env.bash` under root dir
3. Make sure you have access to hugging face or download "openai/clip-vit-large-patch14" from [here](https://huggingface.co/openai/clip-vit-large-patch14) and put it under folder `openai/clip-vit-large-patch14`

## Model Zoo
Put these models under `checkpoint` folder
https://drive.google.com/drive/folders/1HlagN6jVhmC_UbrOAy133LkN4Qgf2Scv?usp=sharing

## Train
Before training, please download the [checkpoint](https://drive.google.com/file/d/1JbJ7tWB15DzCB9pfLKnUHglckumOdUio/view) of the off-the-shelf detector into a folder called `checkpoint/`. 
```
python train.py --class_split 1 --train_data random --save_name pascal_1_random 
```

## Inference
```bash
python test.py --sd_ckpt 'xxx/stable_diffusion.ckpt' \
--grounding_ckpt 'xxx/grounding_module.pth' \
--prompt "a photo of a lion on a mountain top at sunset" \
--category "lion"

```
## Citation
	@article{li2023grounded,
	  title   = {Open-vocabulary Object Segmentation with Diffusion Models},
	  author  = {Li, Ziyi and Zhou, Qinye and Zhang, Xiaoyun and Zhang, Ya and Wang, Yanfeng and Xie, Weidi},
	  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
	  year    = {2023}
	}
	
## Acknowledgements
Many thanks to the code bases from [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [CLIP](https://github.com/openai/CLIP), [taming-transformers](https://github.com/CompVis/taming-transformers), [mmdetection](https://github.com/open-mmlab/mmdetection)
