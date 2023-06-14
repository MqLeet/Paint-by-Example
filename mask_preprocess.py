import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
import copy

import PIL
import torchvision.transforms as transforms
from pathlib import Path
# ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True

# This is the dmd dataset's mask
# path = '/home/duanyuxuan/dms_gen/CIHP_PGN/output/dmd_extract1k_crop768-1024/cihp_parsing_maps/' #图像中人体区域的解析mask，不同的区域（衣服，肢体，脸，背景）解析值不同
# path_edit_mask = '/home/duanyuxuan/dms_gen/CIHP_PGN/output/dmd_extract1k_crop768-1024/edit_mask/' # 脸部+头发 mask
# path_ref_mask = '/home/duanyuxuan/dms_gen/CIHP_PGN/output/dmd_extract1k_crop768-1024/ref_mask/' # 全黑 或许是背景的参考mask？
# path_whole_mask = '/home/duanyuxuan/dms_gen/CIHP_PGN/output/dmd_extract1k_crop768-1024/whole_mask/' # 图像中人体区域的mask， 人体有关的区域均为同一值

# This is the Dreambooth clothing with dmd dataset's mask
# path = '/data/hongyan/CIHP_PGN/output/selected_dmd_512/cihp_parsing_maps/'
# path_edit_mask = '/data/hongyan/CIHP_PGN/output/selected_dmd_512/edit_mask/'
# path_face_mask = '/data/hongyan/CIHP_PGN/output/selected_dmd_512/face_mask/'
# path_ref_mask = '/data/hongyan/CIHP_PGN/output/selected_dmd_512/ref_mask/'
# path_whole_mask = '/data/hongyan/CIHP_PGN/output/selected_dmd_512/whole_mask/'

parser = argparse.ArgumentParser(description='Dataset choosing')
parser.add_argument(
    "--DATASET",
    type=str,
    default="dmd_0_2023-04-03T15-41-47_dmd_0")
args = parser.parse_args()

# This is the dmd_ids
# path = os.path.join('/data/hongyan/CIHP_PGN/output/dmd_ids/', args.DATASET, 'cihp_parsing_maps/')
# path_edit_mask = os.path.join('/data/hongyan/CIHP_PGN/output/dmd_ids/', args.DATASET, 'edit_mask/')
# path_face_mask = os.path.join('/data/hongyan/CIHP_PGN/output/dmd_ids/', args.DATASET, 'face_mask/')
# path_ref_mask = os.path.join('/data/hongyan/CIHP_PGN/output/dmd_ids/', args.DATASET, 'ref_mask/')
# path_whole_mask = os.path.join('/data/hongyan/CIHP_PGN/output/dmd_ids/', args.DATASET, 'whole_mask/')

path = os.path.join('/data/hongyan/CIHP_PGN/output/', args.DATASET, 'cihp_parsing_maps/')
path_edit_mask = os.path.join('/data/hongyan/CIHP_PGN/output/', args.DATASET, 'edit_mask/')
path_face_mask = os.path.join('/data/hongyan/CIHP_PGN/output/', args.DATASET, 'face_mask/')
path_ref_mask = os.path.join('/data/hongyan/CIHP_PGN/output/', args.DATASET, 'ref_mask/')
path_whole_mask = os.path.join('/data/hongyan/CIHP_PGN/output/', args.DATASET, 'whole_mask/')

path_masks = list(Path(path).glob('*'))
for p in [path_edit_mask,path_ref_mask,path_whole_mask, path_face_mask]:
    if not os.path.exists(p):
        os.mkdir(p)

for p in path_masks:
    name = str(p).split('/')[-1]
    if name.split('.')[-1] != 'png':
        continue
    mask=Image.open(p).convert("L")
    mask = np.array(mask)#[None,None]
    mask = 1 - mask.astype(np.float32)/255.0
    # print('h',np.min(mask),np.max(mask))
    mask_edit = copy.deepcopy(mask)
    mask_ref = copy.deepcopy(mask)
    mask_face = copy.deepcopy(mask)
    mask_whole = copy.deepcopy(mask)
    ##
    # 0.2980392: hand
    # 0.5058824: cloth
    # 0.7019608: hair
    # 0.78431374: bozi
    # 0.8509804: face
    # mask[mask < 0.5] = 0
    # mask[mask >= 0.5] = 1
    # mask[mask == 0.8862745] = 0
    # mask[mask != 0] = 1
    mask_edit[mask_edit == 0.8862745] = 0 #face
    mask_edit[mask_edit == 0.7019608] = 0 #hair
    mask_edit[mask_edit != 0] = 1

    mask_face[mask_face == 0.8862745] = 0 #face
    mask_face[mask_face != 0] = 1

    # mask_ref[mask_ref == 0.78431374] = 0 #bozi
    # mask_ref[mask_edit == 0.2980392] = 0 #hand
    mask_ref[mask_ref != 0] = 1

    # print(np.max(mask_whole),np.min(mask_whole))
    # mask_whole[mask_whole == 0.2980392] = 0
    # mask_whole[mask_whole == 0.5058824] = 0
    # mask_whole[mask_whole == 0.7019608] = 0
    # mask_whole[mask_whole == 0.78431374] = 0
    # mask_whole[mask_whole == 0.8509804] = 0
    # mask_whole[mask_whole == 0.8862745] = 0
    # mask_whole[mask_whole!=0] = 1
    # print(np.min(mask_whole),np.max(mask_whole))
    mask_whole[mask_whole!=1] = 0

    mask_edit = np.uint8((1-mask_edit)*255)
    mask_i_edit = Image.fromarray(mask_edit,mode='L')
    mask_i_edit.save(os.path.join(path_edit_mask,name))
                        

    mask_ref = np.uint8((1-mask_ref)*255)
    mask_i_ref = Image.fromarray(mask_ref,mode='L')
    mask_i_ref.save(os.path.join(path_ref_mask,name))

    mask_face = np.uint8((1-mask_face)*255)
    mask_i_face = Image.fromarray(mask_face,mode='L')
    mask_i_face.save(os.path.join(path_face_mask,name))

    mask_whole = np.uint8((1-mask_whole)*255)
    # mask_whole = np.uint8((mask_whole)*255)
    mask_whole = Image.fromarray(mask_whole,mode='L')
    if name.split('.')[0].split('_')[-1]!='vis':
        continue
    else: # 只有读入的图片是vis时，不同部位才会有除了黑白之外的颜色
        name = name.replace('_vis','')
        # mask_whole = mask_whole.resize((832,1088))
        mask_whole.save(os.path.join(path_whole_mask,name))



