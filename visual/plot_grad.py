import glob
import tqdm
import os
import math
import argparse
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = 10000000000
import numpy as np
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import torch
import cv2


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class DivideToScales(object):
    def __init__(self, scale_levels: list, size=None, interpolation=Image.BICUBIC):
        assert len(scale_levels) > 0, "Atleast 1 scale is required. Got: {}".format(scale_levels)

        self.scale_levels = sorted(scale_levels) #[0.25, 0.5, 0.75, 1.0]
        self.num_scales = len(scale_levels)

        if size is not None:
            self.resize = self.get_sizes(scale_levels=scale_levels, size=size)
        else:
            self.resize = None
        self.interpolation = interpolation

    @staticmethod
    def get_sizes(scale_levels, size):
        resize = dict()
        height_1x, width_1x = size
        for sc in scale_levels:
            scaled_h, scaled_w = int(sc * height_1x), int(sc * width_1x)
            # assert height_1x == int(scaled_h / sc), "Scale is not correct. Got: {} and {}".format(height_1x,
            #                                                                                       int(scaled_h / sc))
            # assert width_1x == int(scaled_w / sc), "Scale is not correct. Got: {} and {}".format(width_1x,
            #                                                                                      int(scaled_w / sc))
            resize[sc] = [scaled_h, scaled_w]
        return resize

    @staticmethod
    def get_params(image_sizes, crop_zoom):
        image_width, image_height = image_sizes
        crop_height = int(round(image_height / crop_zoom))
        crop_width = int(round(image_width / crop_zoom))
        return crop_height, crop_width

    def divide_image_to_scales(self, image, mask, scale):
        image = F.resize(image, self.resize[scale], self.interpolation)
        if mask is not None:
             mask = F.resize(mask, self.resize[scale], Image.NEAREST)
        return image, mask

    def get_scales(self, sample: dict):
        image, mask = sample['image'], sample['mask']
        if self.resize is None:
            width, height = _get_image_size(img=image)
            self.resize = self.get_sizes(scale_levels=self.scale_levels, size=(height, width))

        images, masks = [], [] if mask is not None else None
        for i in self.scale_levels:
            im, m = self.divide_image_to_scales(image, mask, i)
            images.append(im)
            if mask is not None:
                masks.append(m)

        return {'image': images, 'mask': masks}

    def __call__(self, sample: dict):
        return self.get_scales(sample=sample)


def calculate_bbox(img_size, resize2, idx):
    """
    Function that return the bounding box of a word given its index
    Args:
        ind: int, ind < number of words

    Returns:
        Bounding box(int[]): [h_low, h_high, w_low, w_high]
    """
    h, w = img_size
    c_w = w // resize2
    c_h = h // resize2
    crop_length = c_w * c_h
    assert idx < crop_length, "Index Out of Bound"

    # [index]: [pad_top, pad_left, pad_right, pad_bottom]
    # top= max((idx % c_h) * (resize2), 0)
    top = max(math.floor(idx / c_w) * resize2, 0)
    # bottom = min(h, (idx % c_h) * resize2+ resize2)
    bottom = min(h, math.floor(idx / c_w) * resize2 + resize2)
    left = max((idx % c_w) * resize2, 0)
    right = min(w, (idx % c_w) * resize2 + resize2)
    # left = max(math.floor(idx / c_h) * resize2, 0)
    # right = min(w, math.floor(idx / c_h) * resize2 + resize2)

    return [top, bottom, left, right]

def resize_image_to_k_crops_size(image, n_crops):
    w, h = image.size
    crop_size_h = int(math.ceil(h / n_crops))
    crop_size_w = int(math.ceil(w / n_crops))

    # transform crop
    # resize crop to fit the crop size
    new_h = n_crops * crop_size_h
    new_w = n_crops * crop_size_w

    image = F.resize(img=image, size=[new_h, new_w], interpolation=Image.BICUBIC)
    return image, (crop_size_w, crop_size_h)


def colormap_grad_to_image_size(image_size, crop_size, grad, n_crops):

    w, h = image_size
    crop_w, crop_h = crop_size
    grad_im = np.array(Image.new('L', size=(w,h)))
    for i in range(n_crops):
        for j in range(n_crops):
            # row by row
            top = i * crop_h
            bottom = top + crop_h
            left = j * crop_w
            right = left + crop_w
            grad_im[top:bottom, left:right] = grad[i*n_crops+j]

    return Image.fromarray(grad_im)

def colormap_to_image(image, cmap='inferno'):
    cm = plt.get_cmap(cmap)
    colored_image = cm(image)

    colored_image = Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))
    return colored_image



def plt_colormap(image, savename, cmap='inferno'):
    plt.figure()
    cm = plt.get_cmap(cmap)
    image = image / 255
    plt.imshow(image, cmap=cm)
    plt.axis('off')
    plt.colorbar()
    plt.savefig(savename)
    plt.close('all')


def parse_img_path_to_dict(data):
    path_dict = {}
    data_txts = glob.glob(os.path.join(data, 'test.txt'))
    assert os.path.exists(data)
    lines = []
    for data_txt in data_txts:
        with open(data_txt, 'r') as f:
            lines.extend([line.rstrip().split(';')[0] for line in f])
    for line in lines:
        assert os.path.exists(line)
        bn = os.path.splitext(os.path.basename(line))[0]
        path_dict[bn] = line
    return path_dict


def find_roi_mask(im_p):
    roi_path = '/projects/patho2/melanoma_diagnosis/ROI/'
    # im_p = '/projects/patho2/melanoma_diagnosis/x10/split/test/4/MP_0312_x10_z0_1.tif'
    im_dir = im_p.split('/test/')[1] # test/4/MP_...
    im_dir = im_dir.split('MP')[0]
    info = os.path.splitext(os.path.basename(im_p))[0].split('_')
    im_ind = info[0] + '_' + info[1]
    if info[2] == 'x10':
        slide_ind = info[-1]
    else:
        slide_ind = info[3]
    roi_mask_p = os.path.join(roi_path, im_dir, im_ind, '{}_{}_mask.png'.format(im_ind, slide_ind))
    if not os.path.exists(roi_mask_p):
        # try to find with glob
        roi_mask_p = glob.glob(os.path.join(roi_path, '*', im_ind, '{}_{}_mask.png'.format(im_ind, slide_ind)))
        if len(roi_mask_p) == 0:
            print('{}_{}_mask.png'.format(im_ind, slide_ind))
            return None
        roi_mask_p = roi_mask_p[0]
    roi_mask = Image.open(roi_mask_p)
    return roi_mask


def plot_roi(roi_image, image, thickness):
    cnts, _ = cv2.findContours(roi_image, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    # xy: [x0, y0, x1, y1]
    image = cv2.drawContours(image, cnts, -1, (0, 255, 0), thickness)
    # draw.rectangle([x, y, x+width, y+height], fill=None, outline=[0, 255, 0], width=thickness)
    return image

def get_concat_h(im1, im2, gray):
    if gray:
        mode = 'L'
        fill = 0
    else:
        mode='RGB'
        fill = (255, 255, 255)
    dst = Image.new(mode, (im1.width + im2.width, max(im1.height, im2.height)), color=fill)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2, gray):
    if gray:
        mode = 'L'
        fill = 0
    else:
        mode='RGB'
        fill = (255, 255, 255)
    dst = Image.new(mode, (max(im1.width, im2.width), im1.height + im2.height), color=fill)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def stitch(all_images, gray=False):
    if len(all_images) < 1:
        pdb.set_trace()
    scales = len(all_images[0])
    stitched = []
    for sc in range(scales):
        at_scale = [images[sc] for images in all_images]
        first_image = at_scale[0]
        w, h = first_image.size
        if w > h: # stitch horizontally
            for i in range(1, len(at_scale)):
                first_image = get_concat_v(first_image, at_scale[i], gray=gray)
        else: # stitch vertically
            for i in range(1, len(at_scale)):
                first_image = get_concat_h(first_image, at_scale[i], gray=gray)
        stitched.append(first_image)
    return stitched
        # stitched --> first_image





def main():
    parser = argparse.ArgumentParser(description='Plot top k crops in whole slide images')
    parser.add_argument('--results-dir', required=True, type=str, default='results', help='results directory location.')
    parser.add_argument('--plot-thickness', type=int, default=20)
    parser.add_argument('--color_map', default='jet', type=str)
    parser.add_argument('--data', default='/projects/patho2/melanoma_diagnosis/x10/experiment_txt/4class_invasive_soft_49_backup/',
                        type=str)
    parser.add_argument('--plot-dir', type=str, default=None)
    parser.add_argument('--scales', default=[1.0], type=float, nargs="+")
    args = parser.parse_args()
    results_dir = args.results_dir
    dir = ['all']
    path_dict = parse_img_path_to_dict(args.data)
    if args.plot_dir is None:
        plot_dir = os.path.join(results_dir, 'summary')
    else:
        plot_dir = args.plot_dir
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    for d in dir:
        assert os.path.exists(os.path.join(results_dir, d)), 'correct and wrong directory not exist in '+ results_dir
        for label_dir in os.listdir(os.path.join(results_dir, d)):
            label_dir = os.path.join(results_dir, d, label_dir)
            if not os.path.isdir(label_dir):
                continue
            print('----processing: {} ----'.format(os.path.basename(label_dir)))
            slices = os.listdir(label_dir)
            cases = {}
            for slice in slices:
                caseid = slice.split('_x10')[0]
                if caseid not in cases:
                    cases[caseid] = [slice]
                else:
                    cases[caseid].append(slice)
            for case in tqdm.tqdm(cases.keys()):
                all_images = []
                all_rois = []
                paths = []
                grads = []
                image_outpath = os.path.join(plot_dir, case)
                # if os.path.exists(image_outpath):
                #     continue
                for case_dir in cases[case]:
                    case_dir = os.path.join(label_dir, case_dir)
                    if not os.path.isdir(case_dir):
                        continue
                    caseid = os.path.basename(case_dir)
                    im_p = path_dict[caseid]
                    paths.append(im_p)
                    original = Image.open(im_p).convert('RGB')
                    # resize to save memory
                    original = original.resize((original.width//4, original.height//4))
                    roi_mask = find_roi_mask(im_p)
                    if roi_mask is None:
                        break
                    roi_mask = roi_mask.resize((roi_mask.width//4, roi_mask.height//4),resample=Image.NEAREST)
                    if original.size != roi_mask.size:
                        print("Image size: {}, Mask size: {}".format(original.size, roi_mask.size))
                    t = DivideToScales(args.scales)
                    sample = t({'image': original, 'mask': roi_mask})
                    image_scales = sample['image']
                    roi_scales = sample['mask']
                    all_images.append(image_scales)
                    all_rois.append(roi_scales)
                    grad_case = {}
                    for scale_dir in os.listdir(case_dir):
                        if not os.path.isdir(os.path.join(case_dir, scale_dir)):
                            continue
                        scale_ind = int(scale_dir.split('scale_')[1])

                        image = image_scales[scale_ind]
                        grad_outpath = os.path.join(case_dir, scale_dir, 'gradient.jpg')
                        overlay_outpath = os.path.join(case_dir, scale_dir, 'gradient_overlay.jpg')
                        roi_outpath = os.path.join(case_dir, scale_dir, 'gradient_roi.jpg')
                        grad = torch.load(os.path.join(case_dir, scale_dir, 'patch_grads.pth'))
                        n_crops = int(math.sqrt(len(grad)))
                        image, crop_size = resize_image_to_k_crops_size(image, n_crops)
                        gray_grad = colormap_grad_to_image_size(image.size, crop_size, grad, n_crops)
                        grad_case[int(scale_ind)] = gray_grad
                    grad_case = [grad_case[i] for i in range(len(grad_case.keys()))]
                    grads.append(grad_case)
                        # colored_grad.save(grad_outpath)
                        # mask_grad = Image.fromarray(np.array(np.array(colored_grad) > 0, dtype=np.uint8)* 255).convert('L')
                    # stitch images

                if not os.path.exists(image_outpath):
                    os.mkdir(image_outpath)

                if len(all_images) < 1:
                    continue
                all_images = stitch(all_images)
                all_rois = stitch(all_rois, gray=True)
                all_grads = stitch(grads, gray=True)
                colored_grads = [colormap_to_image(np.array(grad), args.color_map) for grad in all_grads]
                for sc in range(len(all_images)):
                    plt_colormap(np.array(all_grads[sc]),
                                 os.path.join(image_outpath, 'plt_g_{}.jpg'.format(sc)),
                                 args.color_map)
                    all_images[sc].save(os.path.join(image_outpath, '{}.jpg'.format(sc)))
                    all_rois[sc].save(os.path.join(image_outpath, 'mroi_{}.jpg'.format(sc)))
                    colored_grads[sc].save(os.path.join(image_outpath, 'grads_{}.jpg'.format(sc)))
                    roi_overlay = plot_roi(np.array(all_rois[sc]), np.array(all_images[sc]), args.plot_thickness)
                    Image.fromarray(roi_overlay).save(os.path.join(image_outpath, 'oroi_{}.jpg'.format(sc)))
                    roi_overlay = plot_roi(np.array(all_rois[sc]), np.array(colored_grads[sc]), args.plot_thickness)
                    Image.fromarray(roi_overlay).save(os.path.join(image_outpath, 'groi_{}.jpg'.format(sc)))
                    cg = colored_grads[sc].convert('RGB').resize(all_images[sc].size)
                    overlay = Image.blend(all_images[sc], cg, 0.8)
                    overlay.save(os.path.join(image_outpath, 'goverlay_{}.jpg'.format(sc)))
                # black = gray_grad.convert('RGB')
                # overlay = Image.blend(image, black, 0.3)
                # generate overlay
                # overlay = Image.composite(new_image, image, mask_nongrad)
                # overlay.save(overlay_outpath)

                # Image.fromarray(roi_overlay).save(roi_outpath)


if __name__ == '__main__':
    import pdb
    main()

