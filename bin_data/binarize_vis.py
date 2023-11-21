from config.build import build_dataset
import pdb
import os
from torchvision.utils import save_image
import tqdm
from utilities.print_utilities import *
import torch
import numpy as np
import cv2
import math
from torchvision.transforms import ToPILImage
from dataset.transforms import NormalizeInverse
from visual.visualize import visual_save_grad, compute_case, visual_save_attn, overlay_attn
imagenet_normalization = NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def generate_crop(opts):
    opts['batch_size'] = 1
    opts['binarize'] = True
    dataloaders = build_dataset(opts)
    # one scale at a time
    assert len(opts['resize1_scale']) == 1, 'binarize visualization of 1 scale at a time'
    for dataloader in dataloaders[2:]:
        for k, (image, label, return_label, img_path, _) in tqdm.tqdm(enumerate(dataloader)):
            img_bn = os.path.splitext(os.path.basename(img_path[0]))[0]
            assert len(image) == 1, 'expected 1 scale, got {}'.format(len(image))
            n_crops, channels, width, height = image[0][0].size()
            for i in range(n_crops):
                write_dir = os.path.join(opts['savedir'], img_bn)
                if not os.path.exists(write_dir):
                    os.mkdir(write_dir)
                save_image(image[0][0][i], os.path.join(write_dir, '{}_crop{}.jpg'.format(img_bn, i)))


def calc_gradient(model, multi_data):

    with torch.enable_grad():
        output, output_vis, scale_attn= model(x=multi_data, src_mask=None)
        grads = torch.autograd.grad(output.max(1)[0],
                                   multi_data,
                                   only_inputs=True,
                                   allow_unused=False, create_graph=False)
        # BCHW --> CHW
        return_grads = []
        min_val = 100000
        max_val = -100000
        for grad in grads:
            # grad = grad.squeeze(0)
            grad = grad ** 2
            grad = torch.mean(grad, dim=2)
            grad = torch.sqrt(grad)

            # min-max normalization
            min_v = torch.min(grad)
            max_v = torch.max(grad)
            if min_v < min_val:
                min_val = min_v
            if max_v > max_val:
                max_val = max_v
            return_grads.append(grad)
        return_vis = []
        for grad in return_grads:
            grad = torch.add(grad, -min_val)
            grad = torch.div(grad, max_val - min_val)
            grad *= 255.0
            grad = unify_grad(grad)
            return_vis.append(grad)
    return return_vis


def unify_image(crops):
    bsz, n_crops, C, B_h, B_w = crops.shape
    n_crop = int(math.sqrt(n_crops))
    # [N_B_w * N_B_h x C x B_H x B_W] --> [C x N_B_w * N_B_h x B_H x B_W]
    crops = crops.permute(0, 2, 1, 3, 4)
    # [C x N_B_w * N_B_h x B_H x B_W]  --> [C x N_B_H x N_B_W x B_H x B_W]
    crops = torch.reshape(crops, (bsz, C, n_crop, n_crop, B_h, B_w))
    # [C x N_B_H x N_B_W x B_H x B_W] --> [C x N_B_H x B_H x N_B_W x B_W]
    crops = crops.permute(0, 1, 2, 4, 3, 5)
    # [C x N_B_H x B_H x N_B_W x B_W] --> [C x N_B_H x B_H x W]]
    crops = torch.reshape(crops, (bsz, C, n_crop, B_h, n_crop*B_w))
    # [C x N_B_H x B_H x W] --> [C x H x W]
    crops = torch.reshape(crops, (bsz, C, n_crop *B_h, n_crop * B_w))
    crops = imagenet_normalization({'image': crops[0], 'mask': None})
    # [C x H x W] --> [H x W x C]
    # crops = crops.permute(0, 2, 3, 1)]
    trans = ToPILImage()
    crops = np.array(trans(crops['image']))
    return crops

def unify_grad(crops):
    bsz, n_crops, B_h, B_w = crops.shape
    n_crop = int(math.sqrt(n_crops))
    # [C x N_B_w * N_B_h x B_H x B_W]  --> [C x N_B_H x N_B_W x B_H x B_W]
    crops = torch.reshape(crops, (bsz, n_crop, n_crop, B_h, B_w))
    # [C x N_B_H x N_B_W x B_H x B_W] --> [C x N_B_H x B_H x N_B_W x B_W]
    crops = crops.permute(0, 1, 3, 2, 4)
    # [C x N_B_H x B_H x N_B_W x B_W] --> [C x N_B_H x B_H x W]]
    crops = torch.reshape(crops, (bsz, n_crop, B_h, n_crop*B_w))
    # [C x N_B_H x B_H x W] --> [C x H x W]
    crops = torch.reshape(crops, (bsz, n_crop *B_h, n_crop * B_w))
    crops = crops[0]
    # crops = imagenet_normalization({'image': crops[0], 'mask': None})
    # [C x H x W] --> [H x W x C]
    # crops = crops.permute(0, 2, 3, 1)]
    trans = ToPILImage()
    crops = np.array(trans(crops))
    return crops

def generate_grad(opts):
    from model.msc_model_with_base import MultiScaleAttention
    model = MultiScaleAttention(opts)
    results_list = []
    scores = []
    if opts['resume'] is not None:
        saved_dict = torch.load(opts['resume'])
        print_info_message('Loaded Model')
        model.load_state_dict(saved_dict, strict=False)
        model.eval()
        _,_, test_loader = build_dataset(opts)
        for k, (image, label, return_label, img_path, mask) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
            for data in image:
                data.requires_grad = True
            output, output_vis, scale_attn = model(x=image, src_mask=mask)
            unified_image = [unify_image(torch.clone(im).detach()) for im in image]
            grads = calc_gradient(model, image)
            _, predicted = torch.max(output.data, 1)
            for j in range(len(img_path)):
                results_list.append((img_path[j], int(label[j].item()), int(predicted[j].item())))
            scores.append(output.detach())
            visual_save_grad(predicted, label, grads, unified_image, opts['savedir'], img_path)
        scores = torch.cat(scores, dim=0)
        scores = scores.float()
        probabilities = torch.nn.Sigmoid()(scores)
        probabilities = probabilities.float().numpy()
        compute_case(results_list, probabilities, verbose=True, mode='test',
                     savepath=opts['savedir'], save=False)

def gather_attn_map(attn_wts):
    attn_wts_gathered = []
    for s in range(len(attn_wts)):
        P = attn_wts[s][0].shape[-1]
        # heads = attn_wts[s][0].shape[0]
        attn_map = []
        for l in range(len(attn_wts[s])):
            attn = attn_wts[s][l] # heads x P x P
            attn = attn.mean(dim=-2).detach().numpy() # heads x P 
            attn_map.append(attn)
        attn_wts_gathered.append(attn_map)
    return np.array(attn_wts_gathered)

def generate_attn(opts):
    from model.msc_model import MultiScaleAttention
    model = MultiScaleAttention(opts)
    if opts['resume'] is not None:
        saved_dict = torch.load(opts['resume'])
        saved_dict = {k.replace('module.',''): v for k,v in saved_dict.items()}
        print_info_message('Loaded Model')
        model.load_state_dict(saved_dict, strict=True)
        model.eval()
        _,_,_, test_loader = build_dataset(opts)
        for k, (image, label, return_label, img_path, mask, attn_map) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
            _,_,_, attn_over_layers = model(x=image, src_mask=mask)
            attn_wts = gather_attn_map(attn_over_layers) # num_scales x layers x heads x P

            basename = os.path.splitext(os.path.basename(img_path[0]))[0].split('_')
            basename[2] = 'x2.5'
            slice_path = os.path.join(opts['overlay_img_dir'], '_'.join(basename[:4]), '_'.join(basename)+'.tif')
            try:
                visual_save_attn(attn_wts, os.path.join(opts['savedir'], 'visual_attn'), slice_path)
            
                if opts['attn_guide']:
                    image = cv2.imread(slice_path)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    os.makedirs(os.path.join(opts['savedir'], 'visual_attn'), exist_ok=True)

                    overlayed = overlay_attn(attn_map[0][0].numpy(), image)
                    cv2.imwrite(os.path.join(opts['savedir'], 'visual_attn', '{}_attn_gt.jpg'.format(os.path.basename(slice_path).replace('.tif', ''))), overlayed)
            except:
                print('Fail: {}'.format(slice_path))
