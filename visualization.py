import argparse
from pathlib import Path
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm

#from model_jbl_combine import *
from torchvision.utils import save_image
import itertools
from torch.optim import SGD, Adam
##3d lut

import numpy as np


import os
import argparse
import random
parser = argparse.ArgumentParser()
# 94169
# parser.add_argument('--list_name', type=str, default='total_lut_list.txt')
parser.add_argument('--root_dirs', type=str,default='output_luts')
parser.add_argument('--output_dir', type=str, default='./visualization_hy_results/')
parser.add_argument('--net_output_size', type=int, default=512)

args = parser.parse_args()



args.root_dirs = [
'/data/hongyan/Paint-by-Example/results_hy_test/yawning_and_sleepy/grid/',
'/data/hongyan/PaddleSeg/Matting/results_hy_test/yawning_and_sleepy/alpha/',
'/data/hongyan/PaddleSeg/Matting/results_hy_test/yawning_and_sleepy/rgba/',
]


path = '/data/hongyan/Paint-by-Example/results_hy_test/yawning_and_sleepy/results/'
list_path = sorted(list(os.listdir(path))) #11162, png

visualization_name_naive = ['fusion_3.png','fusion_9.png']


visualization_names = []
for name in visualization_name_naive:
    visualization_names.append(name)

def noCrop_transform(size, crop=False):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size,Image.BICUBIC))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

# content_style_tf = noCrop_transform(size=(args.net_output_size*1,args.net_output_size*5))
content_style_tf_base = noCrop_transform(size=(args.net_output_size*1,args.net_output_size*4))
content_style_tf = noCrop_transform(size=(args.net_output_size*1,args.net_output_size))


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
imgs_all=[]
for i, path_preindex in enumerate(list_path):
    imgs = []
    # print('preindex',path_preindex)
    # if path_preindex not in visualization_names:
    #     continue
    if 0:
        print('selecting')
    else:
        for j,path in enumerate(args.root_dirs):
            path_current = path + path_preindex
            if not os.path.exists(path_current):
                print('not exist',path_current)
            img = Image.open((path_current)).convert('RGB')
            if j==0:
                img = content_style_tf_base(img)
            else:
                img = content_style_tf(img)

            img = torch.unsqueeze(img,dim=0)
            if j==0:
                imgs.append(img[:,:,:,:args.net_output_size])
                imgs.append(img[:,:,:,args.net_output_size:2*args.net_output_size])
                imgs.append(img[:,:,:,2*args.net_output_size:3*args.net_output_size])
                imgs.append(img[:,:,:,3*args.net_output_size:4*args.net_output_size])
            else:
                imgs.append(img)
           
            
            
        imgs = torch.cat(imgs,dim=0)
        output_name = args.output_dir + path_preindex
        print(output_name)
        save_image(imgs, str(output_name),nrow=6)
        # imgs_all.append(imgs)