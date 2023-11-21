import torch
from os.path import basename, dirname, join, splitext, exists
from torch.utils.data import Dataset
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2000000000
import numpy as np
import pickle
import torchvision.transforms as transforms
from dataset.transforms import DivideToCrops, DivideToScales, RandomCrop, Normalize, ToTensor, Resize, Zooming, EvalResize, KCrops
from dataset.transforms import CenterCrop,NumpyToTensor
import pdb

imagenet_normalization = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class MultiScaleAttnDataset(Dataset):
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

        if 'binarized' in self.opts['data']:
            self.patch_num = [np.sqrt(int(self.opts['data'].split('/')[-2])).astype(np.int32)]
        elif 'multi_scale' in self.opts['data']:
            if '3scale' in self.opts['data']:
                self.patch_num = [5, 7, 9]
            else:
                self.patch_num = []
                if '7.5' in self.opts['data']:
                    self.patch_num.append(5)
                if '10' in self.opts['data']:
                    self.patch_num.append(7)
                if '12.5' in self.opts['data']:
                    self.patch_num.append(9)
            
        # check if every image has its corresponding attention map
        img_list = []
        for img_path in self.image_list:
            im_ind = splitext(basename(img_path.split(';')[0]))[0]
            if exists(join(self.opts['attn'], im_ind+'.pickle')):
                img_list.append(img_path)
        self.image_list = img_list

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
        
        with open(join(self.opts['attn'], im_ind+'.pickle'), 'rb') as handle:
            attn_map = pickle.load(handle)
        if np.isnan(attn_map[5]).any():
            # some slices don't have super melanocytes. attn_map will be nan.
            with open(join(self.opts['attn'].replace('super_melanocyte_area', 'melanocyte_num'), im_ind+'.pickle'), 'rb') as handle:
                attn_map = pickle.load(handle) 
        attn_map = [attn_map[p] for p in self.patch_num]

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

        return sample['image'], label, return_label, img_path, sample['mask'] if sample['mask'] is not None else -1, attn_map

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


if __name__ == '__main__':
    import json
    import pdb
    import os.path as path
    from torch.utils.data import DataLoader
    from dataset.sampler.variable_batch_sampler import VariableBatchSampler as VBS

    opts = json.load(open('../example_3scale.json'))
    train_set = MultiScaleAttnDataset(opts=opts,
                                    datasetfile=path.join(opts['data'], 'train.txt'),
                                    datatype='train',
                                    binarized_data=opts['binarize'])
    im_width = opts['resize1'][1]
    im_height = opts['resize1'][0]
    crop_width = opts['resize2'][1]
    crop_height = opts['resize2'][0]
    train_sampler = VBS(n_data_samples=len(train_set),
                        is_training=True,
                        batch_size=opts['batch_size'],
                        im_width=im_width,
                        im_height=im_height,
                        crop_width=crop_width,
                        crop_height=crop_height)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=1,  # Handled inside data sampler
                              num_workers=opts['workers'],
                              pin_memory=False,
                              batch_sampler=train_sampler,
                              persistent_workers=False
                              )
    
    for data in train_loader:
        image, label, return_label, img_path, mask, attn_map = data
        pdb.set_trace()
        print(label)