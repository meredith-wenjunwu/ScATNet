import torch
from os.path import basename, dirname, join, splitext
from torch.utils.data import Dataset
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2000000000
import numpy as np
import torchvision.transforms as transforms
from dataset.transforms import DivideToCrops, DivideToScales, RandomCrop, Normalize, ToTensor, Resize, Zooming, EvalResize, KCrops
from dataset.transforms import CenterCrop,NumpyToTensor
import pdb

imagenet_normalization = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class MultiScaleDataset(Dataset):
    def __init__(self, opts, datasetfile,
                 datatype='train',
                 binarized_data=False,
                 *args, **kwargs):
        self.datatype = datatype
        self.image_case_list = []
        self.maskfolder = opts['mask']
        self.mask_type = opts['mask_type']
        if not isinstance(datasetfile, list):
            datasetfile = [datasetfile]
        self.image_list = []
        for df in datasetfile:
            with open(df, 'r') as f:
                self.image_list.extend([line.rstrip() for line in f])
        self.window_size = opts['resize1']
        self.opts = opts
        self.binarized_data = binarized_data
        self.scale_indices = [int(s) for s in opts['resize1_scale']]


    def training_transforms(self, crop_size):
        msc_transform = DivideToScales if self.opts['transform'] == 'DivideToScale' else Zooming
        return transforms.Compose(
            [
                Resize(max(crop_size)),
                RandomCrop(size=crop_size),
                msc_transform(scale_levels=self.opts['resize1_scale'], size=crop_size),
                ToTensor(),
                DivideToCrops(scale_levels=self.opts['resize2_scale'], crop_size=self.opts['resize2']),
                imagenet_normalization
            ]
        )

    def validation_transforms(self, crop_size):
        msc_transform = DivideToScales if self.opts['transform'] == 'DivideToScale' else Zooming
        return transforms.Compose([
            Resize(min(crop_size)),
            # CenterCrop(size=crop_size),
            msc_transform(scale_levels=self.opts['resize1_scale'], size=crop_size),
            ToTensor(),
            DivideToCrops(scale_levels=self.opts['resize2_scale'], crop_size=self.opts['resize2']),
            imagenet_normalization])

    def test_transforms(self):
        msc_transform = DivideToScales if self.opts['transform'] == 1 else Zooming
        return transforms.Compose([
            ToTensor(),
            msc_transform(scale_levels=self.opts['resize1_scale'], size=self.opts['resize1']),
            DivideToCrops(scale_levels=self.opts['resize2_scale'], crop_size=self.opts['resize2']),
            imagenet_normalization
        ])

    def binary_transform(self):
        msc_transform = DivideToScales
        if self.opts['base_extractor'] == 'mv2':
            return transforms.Compose([
            msc_transform(scale_levels=self.opts['resize1_scale'], size=None, interpolation=Image.BICUBICSS),
            NumpyToTensor(),
            KCrops(scale_levels=self.opts['resize2_scale'], n_crops=self.opts['num_crops']),
        ])
        return transforms.Compose([
            ToTensor(),
            msc_transform(scale_levels=self.opts['resize1_scale'], size=None),
            KCrops(scale_levels=self.opts['resize2_scale'], n_crops=self.opts['num_crops']),
            imagenet_normalization
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, batch_indexes_tup):
        if not isinstance(batch_indexes_tup, int):
            image_h, image_w, idx = batch_indexes_tup
        else:
            image_h, image_w = self.opts['resize1']
            idx = batch_indexes_tup
        img_path = self.image_list[idx]
        img_path, label, select = img_path.split(';')

        use_transform = True
        if img_path.find('.pt') > -1:
            image = torch.load(img_path, map_location='cpu')

            image = [image[i] for i in self.scale_indices]
            use_transform = False
        else:
            image = Image.open(img_path)

        bn = basename(img_path)
        im_ind = splitext(bn)[0]
        label = int(label)
        if self.opts['loss_function'] != 'bce':
            return_label = label
        else:
            return_label = torch.Tensor([float(s) for s in select.split(',')])

        stage = basename(dirname(dirname(img_path)))
        mask = None
        if self.maskfolder is not None:
            mask_path = join(self.maskfolder, stage,
                             im_ind.split('_z0')[0].replace('S2_','') + '_z0','{}.png'.format(im_ind))
            mask = Image.open(mask_path).convert('L')

        #transform_sample = self.transform
        sample = {'image': image, 'mask': mask}

        if use_transform:
            if self.binarized_data:
                transform_sample = self.binary_transform()
                image_w, image_h = image.size
            else:
                if self.datatype.lower() == "train":
                    transform_sample = self.training_transforms(crop_size=(image_h, image_w))
                else:
                    transform_sample = self.validation_transforms(crop_size=(image_h, image_w))
            try:
                sample = transform_sample({'image': image, 'mask': mask}) if transform_sample is not None else sample
            except:
                print(batch_indexes_tup)
                print(img_path)
                exit()

        return sample['image'], label, return_label, img_path, sample['mask'] if sample['mask'] is not None else -1, -1

    def inverse_normalize_image(self, image_tensor, mask, im_ind):
        from os import path
        from dataset.transforms import NormalizeInverse
        from torchvision import transforms
        from torchvision.utils import save_image
        from PIL import Image
        # path --> im_ind
        unnorm_transform = transforms.Compose([NormalizeInverse(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])
        unnorm_image = unnorm_transform(image_tensor)
        save_image(unnorm_image, path.join('/projects/patho1/melanoma_diagnosis/vis/debug/', im_ind))
        if mask is not None:
            mask = mask.squeeze(0)
            mask = np.array(mask)
            mask = Image.fromarray(mask)
            mask.save(path.join('/projects/patho1/melanoma_diagnosis/vis/debug/', 'mask_' + im_ind))
        return
