import os
import argparse
from evaluate import IoU
from seg_module import Segmodule

os.sys.path.append("..")
from sgm.modules.diffusionmodules.openaimodel import clear_feature_dic, get_feature_dic
# from ..sgm.models.diffusion.ddim import DDIMSampler
from scripts.demo.turbo import *
from sgm.util import instantiate_from_config
import torch.optim as optim
import random
import torchvision
from einops import rearrange
from itertools import islice
from tqdm import tqdm, trange
from PIL import Image
from omegaconf import OmegaConf
import numpy as np
import torch.nn as nn
from datetime import datetime
import torch
import PIL
from mmdet.apis import init_detector, inference_detector
# from inference import init_detector, inference_detector
import warnings
from pytorch_lightning import seed_everything
warnings.filterwarnings("ignore")


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


class VOCColorize(object):
    def __init__(self, n):
        self.cmap = color_map(n)

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def color_map(N, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    for name, parameter in model.named_parameters():
        parameter.requires_grad = False

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(h, w)
    print(f"loaded input image of size ({w}, {h}) from {path}")
    # resize to integer multiple of 32
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def load_classes(args):
    print("Loading classes from COCO and PASCAL")
    class_coco = {}
    f = open("configs/data/coco_80_class.txt", "r")
    count = 0
    for line in f.readlines():
        c_name = line.split("\n")[0]
        class_coco[c_name] = count
        count += 1

    pascal_file = "configs/data/VOC/class_split"+str(args.class_split)+".csv"
    class_total = []
    f = open(pascal_file, "r")
    count = 0
    for line in f.readlines():
        count += 1
        class_total.append(line.split(",")[0])
    class_train = class_total[:15]
    class_test = class_total[15:]
    if args.data_mode == 1:
        class_train = class_total[:15]
    elif args.data_mode == 2:
        class_train = class_total[15:]
    return class_train, class_coco


def main(args):
    seed_everything(args.seed)

    class_train, class_coco = load_classes(args)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    config_file = '../src/mmdetection/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
    checkpoint_file = '../checkpoint/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'
    
    pretrain_detector = init_detector(config_file, checkpoint_file, device=device)
    
    seg_module = Segmodule().to(device)
    # state_dic = torch.load('../checkpoint/grounding_module.pth')
    # seg_module.load_state_dict(state_dic)

    version_dict = VERSION2SPECS["SDXL-Turbo"]
    state = init_st(version_dict, load_filter=True)
    model = state["model"] #TODO fp16 or full
    load_model(model)

    sampler = SubstepSampler(
        n_sample_steps=args.n_steps,
        num_steps=1000,
        eta=1.0,
        discretization_config=dict(
            target="sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
        ),
    )
    sampler.noise_sampler = SeededNoise(seed=args.seed)
    
    # sampler.noise_sampler = SeededNoise(seed=args.seed)
    # prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
    # out = sample(model, sampler, H=512, W=512, seed=args.seed, prompt=prompt, filter=state.get("filter"))
    # Image.fromarray(out[0]).save(f'{prompt}.png')

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    save_dir = 'outputs/exps/'
    os.makedirs(save_dir, exist_ok=True)
    ckpt_dir = os.path.join(save_dir, 'ckpts-'+current_time)
    os.makedirs(ckpt_dir, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'logs'))
    os.makedirs(os.path.join(ckpt_dir, 'training'), exist_ok=True)
    
    batch_size = args.n_samples
    learning_rate = 1e-5
    total_iter = 500000
    g_optim = optim.Adam(
        [{"params": seg_module.parameters()},],
        lr=learning_rate
    )
    loss_fn = nn.BCEWithLogitsLoss()
    
    if batch_size > 1:
        print("Model Distributed DataParallel")
        torch.multiprocessing.set_sharing_strategy('file_system')
        seg_module = torch.nn.parallel.DistributedDataParallel(
            module=seg_module, device_ids=[device],
            output_device=device, broadcast_buffers=False)

    print('***********************   begin   **********************************')
    print(f"Start training with maximum {total_iter} iterations.")

    batch_size = args.n_samples
    assert batch_size == 1 # TODO only batch size==1 . see turbo.py line 126 and sample.py do_sample

    # iou = 0
    for j in range(total_iter):
        print('Iter ' + str(j) + '/' + str(total_iter))
        if not args.from_file:
            trainclass = class_train[random.randint(0, len(class_train)-1)]
            prompt = "a photograph of a " + trainclass
            print(f"Iter {j}: prompt--{prompt}")
            assert prompt is not None
            data = [batch_size * [prompt]]
        else:
            raise NotImplementedError
            print(f"reading prompts from {args.from_file}")
            with open(args.from_file, "r") as f:
                data = f.read().splitlines()
                data = list(chunk(data, batch_size))
                
        for prompts in data:
            # class_index = class_coco[trainclass]
            # generate images
            out = sample(
                model, sampler, H=512, W=512, seed=args.seed, 
                prompt=prompts[0], filter=state.get("filter")
            )
            x_sample_list = [out[0]]
            result = inference_detector(pretrain_detector, x_sample_list)
            seg_result_list = []
            for i in range(len(result)):
                seg_result = result[i].pred_instances.masks
                seg_result_list.append(seg_result[0].unsqueeze(0))
            # get class embedding
            class_embedding, uc = sample(
                model, sampler, condition_only=True, H=512, W=512, seed=args.seed, 
                prompt=trainclass, filter=state.get("filter")
            )
            class_embedding = class_embedding['crossattn']
            if class_embedding.size()[1] > 1:
                class_embedding = torch.unsqueeze(class_embedding.mean(1), 1)
            class_embedding = class_embedding.repeat(batch_size, 1, 1)
            # seg_module
            diffusion_features = get_feature_dic()
            total_pred_seg = seg_module(diffusion_features, class_embedding)
            
            loss = []
            for b_index in range(batch_size):
                pred_seg = total_pred_seg[b_index]

                label_pred_prob = torch.sigmoid(pred_seg)
                label_pred_mask = torch.zeros_like(label_pred_prob, dtype=torch.float32)
                label_pred_mask[label_pred_prob > 0.5] = 1
                annotation_pred = label_pred_mask.cpu()

                if len(seg_result_list[b_index]) == 0:
                    print("pretrain detector fail to detect the object in the class:", trainclass)
                else:
                    seg = seg_result_list[b_index]
                    seg = seg.float().cuda() # 1, 512, 512
                    loss.append(loss_fn(pred_seg, seg))
                    annotation_pred_gt = seg.cpu()
                    # iou += IoU(annotation_pred_gt, annotation_pred)
                    # print('iou', IoU(annotation_pred_gt, annotation_pred))
                    viz_tensor2 = torch.cat([annotation_pred_gt, annotation_pred], axis=1)
                    if  j % 100 ==0:
                        dir_path = os.path.join(ckpt_dir, 'training/'+ str(b_index))
                        torchvision.utils.save_image(viz_tensor2, 
                            dir_path +'viz_sample_{0:05d}_seg'.format(j)+trainclass+'.png', 
                            normalize=True, scale_each=True)
                        Image.fromarray(out[0]).save(f'{dir_path + prompts[0]}.png')
                        
            if len(loss) > 0:
                total_loss = 0
                for i in range(len(loss)):
                    total_loss += loss[i]
                total_loss /= batch_size
                g_optim.zero_grad()
                total_loss.backward()
                g_optim.step()

                writer.add_scalar('train/loss', total_loss.item(), global_step=j)
                print("Training step: {0:05d}/{1:05d}, loss: {2:0.4f}".format(j, total_iter, total_loss))
        
        # save checkpoint
        if j % 200 == 0 and j != 0:
            print("Saving latest checkpoint to",ckpt_dir)
            torch.save(seg_module.state_dict(), os.path.join(ckpt_dir, 'checkpoint_latest.pth'))
        if j % 5000 == 0  and j != 0:
            print("Saving latest checkpoint to",ckpt_dir)
            torch.save(seg_module.state_dict(), os.path.join(ckpt_dir, 'checkpoint_'+str(j)+'.pth'))
    
    # print(iou/total_epoch)
    # with open('tmp/ious.txt', "a") as f:
    #     f.write(str(iou/total_epoch)+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_steps",
        type=int,
        default=1,
        help="number of sampling steps",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
        # NOTE: only 8, modify in turbo.sample line 104
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=8,
        help="latent channels",
        # NOTE: only 8, modify in turbo.sample line 105
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoint/stable_diffusion.ckpt",
        help="path to checkpoint of stable-diffusion-v1 model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        help="the save dir name",
        default="exp"
    )
    parser.add_argument(
        "--class_split",
        type=int,
        help="the class split: 1,2,3",
        default=1
    )
    parser.add_argument(
        "--train_data",
        type=str,
        help="the type of training data: single, two, random",
        default="single"
    )
    parser.add_argument(
        "--data_mode",
        type=int,
        default=1,
        help="which data split",
    )

    opt = parser.parse_args()
    main(opt)
