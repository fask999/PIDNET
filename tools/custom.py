# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import glob
import argparse
import cv2
import os
import numpy as np
import _init_paths
import models
import torch
import torch.nn.functional as F
from PIL import Image
from configs import config, update_config
from utils.ablation_config import (register_ablation_config, get_model_kwargs,
                                   get_boundary_loss_kwargs, get_pretrain_source)
from utils.checkpoint import load_model_checkpoint

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = [(128, 64,128),
             (244, 35,232),
             ( 70, 70, 70),
             (102,102,156),
             (190,153,153),
             (153,153,153),
             (250,170, 30),
             (220,220,  0),
             (107,142, 35),
             (152,251,152),
             ( 70,130,180),
             (220, 20, 60),
             (255,  0,  0),
             (  0,  0,142),
             (  0,  0, 70),
             (  0, 60,100),
             (  0, 80,100),
             (  0,  0,230),
             (119, 11, 32)]

def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    
    parser.add_argument('--a', help='pidnet-s, pidnet-m or pidnet-l', default='pidnet-l', type=str)
    parser.add_argument('--c', help='cityscapes pretrained or not', type=bool, default=True)
    parser.add_argument('--p', help='checkpoint path (L experiments can use TEST.MODEL_FILE)', type=str)
    parser.add_argument('--r', help='root or dir for input images', default='../samples/', type=str)
    parser.add_argument('--t', help='the format of input images (.jpg, .png, ...)', default='.png', type=str)     
    parser.add_argument('--cfg', help='experiment config file', default=None, type=str)
    parser.add_argument('--output-dir', help='directory for predicted images', default=None, type=str)
    parser.add_argument('opts', help='config overrides', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.cfg:
        register_ablation_config(config)
        update_config(config, args)
        get_model_kwargs(config)
        get_boundary_loss_kwargs(config)
        if get_pretrain_source(config) != 'legacy':
            if args.p is None:
                args.p = config.TEST.MODEL_FILE
            if not args.p.strip():
                parser.error('PIDNet-L experiment prediction requires --p or TEST.MODEL_FILE')
    elif args.opts:
        parser.error('config overrides require --cfg')

    if args.p is None:
        args.p = '../pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt'
    return args

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

def load_pretrained(model, pretrained):
    count = load_model_checkpoint(model, pretrained, strict=True, prediction=True)
    print('Loaded {} model tensors'.format(count))
    return model

if __name__ == '__main__':
    args = parse_args()
    images_list = glob.glob(os.path.join(args.r, '*' + args.t))
    sv_path = args.output_dir or os.path.join(args.r, 'outputs')
    
    if args.cfg:
        model = models.pidnet.get_pred_model(config.MODEL.NAME, config.DATASET.NUM_CLASSES,
                                              **get_model_kwargs(config))
    else:
        model = models.pidnet.get_pred_model(args.a, 19 if args.c else 11)
    model = load_pretrained(model, args.p).cuda()
    model.eval()
    with torch.no_grad():
        for img_path in images_list:
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            sv_img = np.zeros_like(img).astype(np.uint8)
            img = input_transform(img)
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            pred = model(img)
            pred = F.interpolate(pred, size=img.size()[-2:], 
                                 mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            
            for i, color in enumerate(color_map):
                for j in range(3):
                    sv_img[:,:,j][pred==i] = color_map[i][j]
            sv_img = Image.fromarray(sv_img)
            
            if not os.path.exists(sv_path):
                os.makedirs(sv_path)
            sv_img.save(os.path.join(sv_path, img_name))
