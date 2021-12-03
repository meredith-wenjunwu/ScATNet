from torchvision.transforms import functional as F
from utilities import functional as F_local
import torch
from PIL import Image
import numpy as np
import numbers
import collections
import sys
import random
import math
import pdb

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


# class SlideWindow(object):
#     '''
#     Returns a list of cropped images
#     '''
#     def __init__(self, size, step=0):
#         if isinstance(size, (int, float)):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#         self.unfold = torch.nn.Unfold(self.size, stride=step)
#
#     def __call__(self, sample: dict):
#         image, mask = sample['image'], sample['mask']
#         image = image.unsqueeze(0)
#         crops = self.unfold(image)
#         crops = crops.reshape(3, self.size[0], self.size[1], -1).permute(3, 0, 1, 2)
#         return {'image': crops, 'mask': mask}


class NumpyToTensor(object):
    def __init__(self):
        super(NumpyToTensor, self).__init__()
        self.eps = 1e-9
    def __call__(self, sample: dict):
        # img: np.ndarray, target: Optional[np.ndarray] = None) -> (Tensor, Optional[Tensor]
        image, targets = sample['image'], sample['mask']
        return_image = []
        return_mask = [] if targets is not None else None
        for i in range(len(image)):
            img = image[i]
            target = targets[i] if targets is not None else None
            if F._is_pil_image(img):
                img_rgb = np.array(img)
                # convert RGB to BGR
                img = img_rgb[:, :, ::-1]
            # HWC --> CHW
            img = img.transpose(2, 0, 1)
            # numpy to tensor
            img_tensor = torch.from_numpy(img.copy()).float()
            if target is not None:
                target_tensor = torch.from_numpy(target).long()
                return_mask.append(target_tensor)
            return_image.append(img_tensor)
        return {'image': return_image, 'mask':return_mask}


class KCrops(object):
    def __init__(self, scale_levels: list, n_crops=7):
        self.n_crops_h = n_crops
        self.n_crops_w = n_crops

        self.scales = scale_levels

    def divide_image_to_crops(self, image):
        # resize image into small crops
        channel, h, w = image.shape
        crop_size_h = int(math.ceil(h / self.n_crops_h))
        crop_size_w = int(math.ceil(w / self.n_crops_w))

        # transform crop
        # resize crop to fit the crop size
        new_h = self.n_crops_h * crop_size_h
        new_w = self.n_crops_w * crop_size_w

        # transform to crops
        ## Image to BAGS
        image = F.resize(img=image, size=[new_h, new_w], interpolation=Image.BICUBIC)
        # [C x N_B_H x B_H x W]] --> [C x N_B_H x B_H x N_B_W x B_W]
        crops = torch.reshape(image, (channel, self.n_crops_h, crop_size_h, self.n_crops_w, crop_size_w))
        # [C x N_B_H x B_H x N_B_W x B_W] --> [C x N_B_H x N_B_W x B_H x B_W]
        crops = crops.permute(0, 1, 3, 2, 4)

        # '''
        # Preserve dimensionality, move to forward loop
        # '''
        # #[C x N_B_H x N_B_W x B_H x B_W]--> [C x N_B_w * N_B_h x B_H x B_W]
        crops = torch.reshape(crops, (channel, self.n_crops_h * self.n_crops_w, crop_size_h, crop_size_w))
        # #[C x N_B_w * N_B_h x B_H x B_W] --> [N_B_w * N_B_h x C x B_H x B_W]
        crops = crops.permute(1, 0, 2, 3)
        return crops

    def __call__(self, sample: dict):
        image, mask = sample['image'], sample['mask']
        images = [self.divide_image_to_crops(im) for im, sc in zip(image, self.scales)]

        masks = None
        if mask is not None:
            masks = [self.divide_image_to_crops(m) for m, sc in zip(mask, self.scales)]

        return {'image': images, 'mask': masks}


class DivideToCrops(object):
    def __init__(self, scale_levels: list, crop_size):
        if isinstance(crop_size, (int, float)):
            self.size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
        #assert 1.0 in scale_levels
        self.crop_sizes = dict()
        for sc in scale_levels:
            scaled_size = [int(sc * c_size) for c_size in crop_size]
            assert crop_size[0] == int(scaled_size[0] / sc) and crop_size[1] == int(scaled_size[1] / sc), "Scale is not correct. Got: {} and {}".format(crop_size, int(scaled_size / sc))
            self.crop_sizes[sc] = scaled_size
        self.scales = scale_levels

    @staticmethod
    def divide_image_to_crops(image, crop_size):
        # resize image into small crops
        channel, h, w = image.shape
        # transform crop
        # resize crop to fit the crop size
        new_h = (h // crop_size[0]) * crop_size[0]
        new_w = (w // crop_size[1]) * crop_size[1]

        n_crops_h = new_h // crop_size[0]
        n_crops_w = new_w // crop_size[1]

        # transform to crops
        ## Image to BAGS
        image = image[:, :new_h, :new_w]
        # [C x N_B_H x B_H x W]] --> [C x N_B_H x B_H x N_B_W x B_W]
        crops = torch.reshape(image, (channel, n_crops_h, crop_size[0], n_crops_w, crop_size[1]))
        # [C x N_B_H x B_H x N_B_W x B_W] --> [C x N_B_H x N_B_W x B_H x B_W]
        crops = crops.permute(0, 1, 3, 2, 4)

        # '''
        # Preserve dimensionality, move to forward loop
        # '''
        # #[C x N_B_H x N_B_W x B_H x B_W]--> [C x N_B_w * N_B_h x B_H x B_W]
        crops = torch.reshape(crops, (channel, n_crops_w * n_crops_h, crop_size[0], crop_size[1]))
        # #[C x N_B_w * N_B_h x B_H x B_W] --> [N_B_w * N_B_h x C x B_H x B_W]
        crops = crops.permute(1, 0, 2, 3)

        return crops

    def __call__(self, sample: dict):
        image, mask = sample['image'], sample['mask']
        images = [self.divide_image_to_crops(im, crop_size=self.crop_sizes[sc]) for im, sc in zip(image, self.scales)]

        masks = None
        if mask is not None:
            masks = [self.divide_image_to_crops(m, crop_size=self.crop_sizes[sc]) for m, sc in zip(mask, self.scales)]

        return {'image': images, 'mask': masks}


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
             mask = F.resize(image, self.resize[scale], Image.NEAREST)
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


class Zooming(DivideToScales):
    def __init__(self, scale_levels: list, size, interpolation=Image.BILINEAR):
        super(Zooming, self).__init__(scale_levels=scale_levels, size=size, interpolation=interpolation)
        assert len(scale_levels) > 0, "Atleast 1 scale is required. Got: {}".format(scale_levels)

    @staticmethod
    def get_params(image_sizes, crop_zoom):
        image_width, image_height = image_sizes
        crop_height = int(round(image_height / crop_zoom))
        crop_width = int(round(image_width / crop_zoom))

        return crop_height, crop_width

    def divide_image_to_scales(self, image, mask, scale):
        image = F.resize(image, self.resize[scale], self.interpolation)
        if mask is not None:
            mask = F.resize(image, self.resize[scale], Image.NEAREST)
        return image, mask

    def __call__(self, sample: dict):
        # scaled images
        samples = self.get_scales(sample=sample)

        # minimum scale size
        center_crop_size = self.resize[min(self.scale_levels)]

        # center crops
        center_crop_images, center_crop_masks = [], []

        for idx, img in enumerate(samples['image']):
            mask_img = samples['mask'][idx] if samples['mask'] is not None else None
            res = CenterCrop(size=center_crop_size)({'image': img, 'mask': mask_img})
            center_crop_images.append(res['image'])
            if res['mask'] is not None:
                center_crop_masks.append(res['mask'])

        return {'image': center_crop_images, 'mask': center_crop_masks if len(center_crop_masks) > 0 else None}


class FilterBgMask(object):
    def __init__(self, threshold, reverse=False):
        self.threshold = threshold
        self.reverse = reverse

    def __call__(self, sample):
        def filter_bg_crops_in_mask(mask):
            if mask is None:
                return mask
            #[N_B_w * N_B_h x C x B_H x B_W]
            n_crops, channel, crop_h, crop_w = mask.shape
            mask_to_keep = []
            for i in range(n_crops):
                crop = mask[i] // 255
                if self.reverse:
                    if torch.sum(crop) < self.threshold * crop_h * crop_w:
                        mask_to_keep.append(1)
                    else:
                        mask_to_keep.append(0)
                else:
                    if torch.sum(crop) > self.threshold * crop_h * crop_w:
                        mask_to_keep.append(1)
                    else:
                        mask_to_keep.append(0)
            return torch.tensor(mask_to_keep)
        if type(sample) is dict:
            image, mask = sample['image'], sample['mask']
            return {'image': image, 'mask':[filter_bg_crops_in_mask(m) for m in mask]}
        else:
            return sample


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class PadCenterCrop(object):
    def __init__(self, size, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.pad_if_needed = pad_if_needed
        self.padding_mode = padding_mode
        self.fill = fill

    def __call__(self, sample):
        if type(sample) is dict:
            img, mask = sample['image'], sample['mask']
        else:
            img = sample
        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            mask = F.pad(mask, (self.size[1] - mask.size[0], 0), 0, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            mask = F.pad(mask, (0, self.size[0] - mask.size[1]), 0, self.padding_mode)

        if type(sample) is dict:
            return {'image': F.center_crop(img, self.size),
                    'mask': F.center_crop(mask, self.size)}
        else:
            return F.center_crop(img, self.size)


class CenterCrop(object):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.
        Returns:
            PIL Image or Tensor: Cropped image.
        """
        img, mask = sample['image'], sample['mask']
        img = F.center_crop(img, self.size)
        if mask is not None:
            mask = F.center_crop(mask, self.size)
        return {'image': img, 'mask': mask}

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)



class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample: dict):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        image, mask = sample['image'], sample['mask']

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            if mask is not None:
                mask = F.pad(mask, self.padding, self.fill, self.padding_mode)
        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            #print('in transform: mask size {}, {}'.format(mask.size[0], mask.size[1]))
            if mask is not None:
                mask = F.pad(mask, (self.size[1] - mask.size[0], 0), 0, self.padding_mode)

        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            if mask is not None:
                mask = F.pad(mask, (0, self.size[0] - mask.size[1]), 0, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        image = F.crop(image, i, j, h, w)
        if mask is not None:
            mask = F.crop(mask, i, j, h, w)
        return {'image': image, 'mask': mask}

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands. This option is only available for ``pillow>=5.2.0``.
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """

    def __init__(self, degrees, resample=False,
                 expand=False, center=None, fill=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """
        angle = self.get_params(self.degrees)
        if type(sample) is dict:
            image, mask = sample['image'], sample['mask']
            if isinstance(image, list):
                return {'image': [F_local.rotate(im, angle, self.resample,
                                                self.expand, self.center, self.fill) for im in image],
                       'mask': [F_local.rotate(m, angle, self.resample,
                                              self.expand, self.center, self.fill) for m in mask]}
            else:
                return {'image': F_local.rotate(image, angle, self.resample,
                                          self.expand, self.center, self.fill),
                        'mask': F_local.rotate(mask, angle, self.resample,
                                         self.expand, self.center, self.fill)}
        else:
            if isinstance(sample, list):
                return [F_local.rotate(s, angle, self.resample,
                                       self.expand, self.center, self.fill) for s in sample]
            else:
                return F_local.rotate(sample, angle, self.resample,
                                self.expand, self.center, self.fill)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        image, mask = sample['image'], sample['mask']
        if random.random() < self.p:
            if not isinstance(image, list):
                return {'image': F.vflip(image), 'mask': F.vflip(mask) if mask is not None else None}
            return {'image': [F.vflip(im) for im in image],
                    'mask': [F.vflip(m) for m in mask] if mask is not None else None}
        else:
            return sample


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if type(sample) is dict:
            image, mask = sample['image'], sample['mask']
            if random.random() < self.p:
                return {'image': F.hflip(image), 'mask': F.hflip(mask)}
            else:
                return sample
        else:
            if random.random() < self.p:
                return F.hflip(sample)
            return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


'''
TODO: make another class of Random Flip (Horizontal, Vertical, no flip)
'''


class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if type(sample) is dict:
            image, mask = sample['image'], sample['mask']
            if random.random() <= self.p:
                flip = random.choice([F.hflip, F.vflip])
                return {'image': flip(image), 'mask': flip(mask)}
            else:
                return sample
        else:
            if random.random() < self.p:
                flip = random.choice([F.hflip, F.vflip])
                return flip(sample)
            return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        if type(sample) is dict:
            image, mask = sample['image'], sample['mask']
            i, j, h, w = self.get_params(image, self.scale, self.ratio)
            return {'image': F.resized_crop(image, i, j, h, w, self.size, self.interpolation),
                   'mask': F.resized_crop(mask, i, j, h, w, self.size, Image.NEAREST)}
        else:
            i, j, h, w = self.get_params(sample, self.scale, self.ratio)
            return F.resized_crop(sample, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string



class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, sample: dict):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        image, mask = sample['image'], sample['mask']
        image_tensor = [F.to_tensor(im) for im in image] if isinstance(image, list) else F.to_tensor(image)
        mask_tensor = None
        if mask is not None:
            mask_tensor = [torch.ByteTensor(np.array(m)).unsqueeze(0) for m in mask] if isinstance(mask, Iterable) else F.to_tensor(mask)

        return {'image': image_tensor, 'mask': mask_tensor}

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Resize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        image, mask = sample['image'], sample['mask']
        if isinstance(image, list):
            image = [F.resize(im, self.size, self.interpolation) for im in image]
            if mask is not None:
                mask = [F.resize(m, self.size, Image.NEAREST) for m in mask]
        else:
            image = F.resize(image, self.size, self.interpolation)
            if mask is not None:
                mask = F.resize(mask, self.size, Image.NEAREST)
        return {'image': image, 'mask': mask}

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class EvalResize(object):
    """Resize the input PIL Image to match the longest side.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        image, mask = sample['image'], sample['mask']
        w, h = _get_image_size(image)
        s = max(self.size) if isinstance(self.size, Iterable) else self.size
        im_s = max(image.size)
        ratio = im_s/s
        image = F.resize(image, (int(h/ratio), int(w/ratio)), self.interpolation)
        if mask is not None:
            mask = F.resize(mask, (int(h*ratio), int(w*ratio)), Image.NEAREST)
        return {'image': image, 'mask': mask}

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        image, mask = sample['image'], sample['mask']
        image = [F.normalize(im, self.mean, self.std, self.inplace) for im in image]
        return {'image': image, 'mask': mask}

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class NormalizeInverse(Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, sample):
        return super().__call__(sample)

