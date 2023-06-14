from torchvision.transforms import Resize
import clip
from transformers import AutoFeatureExtractor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
import copy
from pathlib import Path
import argparse
import os
import sys
import glob
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

sys.path.append('/data/hongyan/Paint-by-Example/')


wm = "Paint-by-Example"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(
    safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open(
            "assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(
        numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(
        images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


def fillHole(im_in):
    im_floodfill = im_in.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_in | im_floodfill_inv
    return im_out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./results_hy/"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
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
        default=768,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_imgs",
        type=int,
        default=100,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given reference image. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="evaluate at this precision",
        default=""
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        help="evaluate at this precision",
        default=""
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        help="evaluate at this precision",
        default=""
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        help="evaluate at this precision",
        default=""
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        help="evaluate at this precision",
        default=""
    )

    parser.add_argument(
        "--reference_dir",
        type=str,
        help="evaluate at this precision",
        default=""
    )

    opt = parser.parse_args()

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    sample_path = os.path.join(outpath, "source")
    result_path = os.path.join(outpath, "results")
    mask_path = os.path.join(outpath, "masks")
    inpaint_path = os.path.join(outpath, "inpaints")
    grid_path = os.path.join(outpath, "grid")
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(grid_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(inpaint_path, exist_ok=True)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn(
            [opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            images_paths = list(Path(opt.image_dir).glob("*"))
            with model.ema_scope():
                reference_paths = list(
                    Path(opt.reference_dir).glob("*.jpg"))  # 生成的yawn-sleepy表情
                for n_r, reference_path in enumerate(reference_paths):
                    ref_p = Image.open(reference_path).convert(
                        "RGB").resize((224, 224))
                    ref_tensor = get_tensor_clip()(ref_p)
                    ref_tensor = ref_tensor.unsqueeze(0)

                    for n_i, path in enumerate(images_paths):
                        # if n_i > 10:
                        #     break
                        # .split('.')[0].split('_')[-1]
                        img_name = str(path).split('/')[-1]
                        save_name = 'ref' + str(reference_path).split('/')[-1].split(
                            '.')[0] + '_' + 'source_' + img_name.split('.')[0] + '.png'
                        print(os.path.exists(opt.image_dir + img_name),
                              opt.image_dir + img_name)
                        img_p = Image.open(
                            opt.image_dir + img_name).convert("RGB")
                        image_tensor = get_tensor()(img_p)
                        image_tensor = image_tensor.unsqueeze(0)
                        # mask=Image.open(opt.mask_dir + img_name.replace('.jpg','_vis.png')).convert("L")
                        # mask = cv2.imread(opt.mask_dir + img_name.replace('.jpg','_vis.png'), cv2.IMREAD_GRAYSCALE) # 如果是段宇轩学长提供的dmd数据集的原图，后缀是.jpg 修改时间： 2023.4.10

                        # 马千里进行裁剪之后得到的后缀是png
                        mask = cv2.imread(
                            opt.mask_dir + img_name.replace('.png', '_vis.png'), cv2.IMREAD_GRAYSCALE)

                        mask = fillHole(mask)
                        mask = np.array(mask)[None, None]
                        mask = 1 - mask.astype(np.float32)/255.0
                        # generate edit area from DYX results
                        # face: blue
                        # hair: red 0.2980392
                        # print(np.max(mask),np.min(mask))
                        # 0.2980392: hand
                        # 0.5058824: cloth
                        # 0.7019608: hair
                        # 0.78431374: bozi
                        # 0.8509804: face
                        # [0.2980392  0.45098037 0.5058824  0.7019608  0.78431374 0.8509804 0.8862745  1.        ]
                        # mask[mask < 0.5] = 0
                        # mask[mask >= 0.5] = 1
                        # mask[mask == 0.8862745] = 0
                        # mask[mask != 0] = 1

                        # ### select edit area
                        # mask[mask == 0.8862745] = 0 #face
                        # mask[mask == 0.7019608] = 0 #hair
                        # # mask[mask == 0.78431374] = 0 #bozi
                        # mask[mask != 0] = 1

                        # ###remove hole, like glasses
                        # mask_inverse = ((1-copy.deepcopy(mask))* 255).astype(np.uint8)
                        # mask_inverse = fillHole(mask_inverse)
                        # mask=(1-mask_inverse/255).astype(np.float32)

                        mask_tensor = torch.from_numpy(mask)
                        inpaint_image = image_tensor*mask_tensor
                        test_model_kwargs = {}
                        test_model_kwargs['inpaint_mask'] = Resize(
                            [opt.H, opt.W])(mask_tensor.to(device))
                        test_model_kwargs['inpaint_image'] = Resize(
                            [opt.H, opt.W])(inpaint_image.to(device))
                        ref_tensor = ref_tensor.to(device)
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.learnable_vector
                        c = model.get_learned_conditioning(
                            ref_tensor.to(torch.float16))
                        c = model.proj_out(c)
                        inpaint_mask = test_model_kwargs['inpaint_mask']
                        z_inpaint = model.encode_first_stage(
                            test_model_kwargs['inpaint_image'])
                        z_inpaint = model.get_first_stage_encoding(
                            z_inpaint).detach()
                        test_model_kwargs['inpaint_image'] = z_inpaint
                        test_model_kwargs['inpaint_mask'] = Resize(
                            [z_inpaint.shape[-2], z_inpaint.shape[-1]])(test_model_kwargs['inpaint_mask'])

                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code,
                                                         test_model_kwargs=test_model_kwargs)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image, has_nsfw_concept = check_safety(
                            x_samples_ddim)
                        x_checked_image = x_samples_ddim
                        x_checked_image_torch = torch.from_numpy(
                            x_checked_image).permute(0, 3, 1, 2)

                        def un_norm(x):
                            return (x+1.0)/2.0

                        def un_norm_clip(x):
                            x[0, :, :] = x[0, :, :] * 0.26862954 + 0.48145466
                            x[1, :, :] = x[1, :, :] * 0.26130258 + 0.4578275
                            x[2, :, :] = x[2, :, :] * 0.27577711 + 0.40821073
                            return x

                        if not opt.skip_save:
                            for i, x_sample in enumerate(x_checked_image_torch):
                                all_img = []
                                all_img.append(
                                    un_norm(Resize([opt.H, opt.W])(image_tensor[i]).cpu()))
                                all_img.append(
                                    un_norm(Resize([opt.H, opt.W])(inpaint_image[i]).cpu()))
                                ref_img = ref_tensor
                                ref_img = Resize([opt.H, opt.W])(ref_img)
                                all_img.append(un_norm_clip(ref_img[i]).cpu())
                                all_img.append(x_sample)
                                grid = torch.stack(all_img, 0)
                                grid = make_grid(grid)
                                grid = 255. * \
                                    rearrange(
                                        grid, 'c h w -> h w c').cpu().numpy()
                                img = Image.fromarray(grid.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(grid_path, save_name))

                                x_sample = 255. * \
                                    rearrange(x_sample.cpu().numpy(),
                                              'c h w -> h w c')
                                img = Image.fromarray(
                                    x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                # remove failure case
                                print(save_name, np.mean(np.array(img)))
                                if np.mean(np.array(img)) < 1:
                                    continue
                                img.save(os.path.join(result_path, save_name))

                                mask_save = 255. * \
                                    rearrange(
                                        un_norm(inpaint_mask[i]).cpu(), 'c h w -> h w c').numpy()
                                mask_save = cv2.cvtColor(
                                    mask_save, cv2.COLOR_GRAY2RGB)
                                mask_save = Image.fromarray(
                                    mask_save.astype(np.uint8))
                                # mask_save.save(os.path.join(sample_path, "mask.png"))
                                mask_save.save(os.path.join(
                                    mask_path, save_name))

                                GT_img = 255. * \
                                    rearrange(
                                        all_img[0], 'c h w -> h w c').numpy()
                                GT_img = Image.fromarray(
                                    GT_img.astype(np.uint8))
                                # GT_img.save(os.path.join(sample_path, "GT.png"))
                                inpaint_img = 255. * \
                                    rearrange(
                                        all_img[1], 'c h w -> h w c').numpy()
                                inpaint_img = Image.fromarray(
                                    inpaint_img.astype(np.uint8))
                                # inpaint_img.save(os.path.join(sample_path, "inpaint.png"))
                                inpaint_img.save(os.path.join(
                                    inpaint_path, save_name))

                                ref_img = 255. * \
                                    rearrange(
                                        all_img[2], 'c h w -> h w c').numpy()
                                ref_img = Image.fromarray(
                                    ref_img.astype(np.uint8))
                                ref_img.save(os.path.join(
                                    sample_path, "ref.png"))

        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
              f" \nEnjoy.")


if __name__ == "__main__":
    main()
