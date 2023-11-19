from dataset.multi_scale_dataset import MultiScaleDataset
from dataset.multi_scale_attn_dataset import MultiScaleAttnDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from os import path
from typing import Optional, List
from config.build import seed_everything
import pdb
from dataset.sampler.variable_batch_sampler import VariableBatchSampler as VBS

def create_datasets(opts):
    if not opts['attn_guide']:
        train_set = MultiScaleDataset(opts=opts,
                                    datasetfile=path.join(opts['data'], 'train.txt'),
                                    datatype='train',
                                    binarized_data=opts['binarize'])

        val_set = MultiScaleDataset(opts=opts,
                                    datasetfile=path.join(opts['data'], 'valid.txt'),
                                    datatype='valid',
                                    binarized_data=opts['binarize'])
        test_set = MultiScaleDataset(opts=opts,
                                    datasetfile=path.join(opts['data'], 'test.txt'),
                                    datatype='valid',
                                    binarized_data=opts['binarize'])
        merge_train_val_set = MultiScaleDataset(opts=opts, 
                                                datasetfile=[path.join(opts['data'], 'train.txt'), path.join(opts['data'], 'valid.txt')],
                                                datatype='train', 
                                                binarized_data=opts['binarize'])
    else:
        train_set = MultiScaleAttnDataset(opts=opts,
                                datasetfile=path.join(opts['data'], 'train.txt'),
                                datatype='train',
                                binarized_data=opts['binarize'])

        val_set = MultiScaleAttnDataset(opts=opts,
                                    datasetfile=path.join(opts['data'], 'valid.txt'),
                                    datatype='valid',
                                    binarized_data=opts['binarize'])
        test_set = MultiScaleAttnDataset(opts=opts,
                                    datasetfile=path.join(opts['data'], 'test.txt'),
                                    datatype='valid',
                                    binarized_data=opts['binarize'])
        merge_train_val_set = MultiScaleAttnDataset(opts=opts, 
                                                    datasetfile=[path.join(opts['data'], 'train.txt'), path.join(opts['data'], 'valid.txt')],
                                                    datatype='train', 
                                                    binarized_data=opts['binarize'])

    return train_set, val_set, merge_train_val_set, test_set


def create_dataloader(train_set, val_set, merge_train_val_set, test_set, opts):
    seed_everything(opts)
    #train_sampler = SubsetRandomSampler(list(range(len(train_set))))

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

    #valid_sampler = SubsetRandomSampler(list(range(len(val_set))))

    valid_sampler = VBS(n_data_samples=len(val_set),
                        is_training=False,
                        batch_size=opts['batch_size'],
                        im_width=im_width,
                        im_height=im_height,
                        crop_width=crop_width,
                        crop_height=crop_height
                        )
    test_sampler = VBS(n_data_samples=len(test_set), is_training=False, batch_size=1,
                       im_width=im_width, im_height=im_height, crop_width=crop_width, crop_height=crop_height)
    
    train_valid_sampler = VBS(n_data_samples=len(merge_train_val_set),
                        is_training=True,
                        batch_size=opts['batch_size'],
                        im_width=im_width,
                        im_height=im_height,
                        crop_width=crop_width,
                        crop_height=crop_height)
    """
    train_loader = DataLoader(train_set, batch_size=opts['batch_size'],
                              sampler=train_sampler,
                              num_workers=opts['workers'],
                              pin_memory = False)
    """

    train_loader = DataLoader(dataset=train_set,
                              batch_size=1,  # Handled inside data sampler
                              num_workers=opts['workers'],
                              pin_memory=False,
                              batch_sampler=train_sampler,
                              persistent_workers=False
                              )

    valid_loader = DataLoader(dataset=val_set,
                              batch_size=1,  # Handled inside data sampler
                              num_workers=opts['workers'],
                              pin_memory=False,
                              batch_sampler=valid_sampler,
                              persistent_workers=False
                              )
    
    train_valid_loader = DataLoader(dataset=merge_train_val_set,
                              batch_size=1,  # Handled inside data sampler
                              num_workers=opts['workers'],
                              pin_memory=False,
                              batch_sampler=train_valid_sampler,
                              persistent_workers=False
                              )
    """
    valid_loader = DataLoader(val_set, batch_size=opts['batch_size'],
                              sampler=valid_sampler,
                              num_workers=opts['workers'],
                              pin_memory = False)

    test_loader = DataLoader(test_set, batch_size=opts['batch_size'],
                             num_workers=opts['workers'],
                             pin_memory = False)
    """

    test_loader = DataLoader(dataset=test_set,
                              batch_size=1,  # Handled inside data sampler
                              num_workers=opts['workers'],
                              pin_memory=False,
                              batch_sampler=test_sampler,
                              persistent_workers=False
                              )

    return train_loader, valid_loader, train_valid_loader, test_loader


def sliding_dataloader(dataset, index):
    return DataLoader(dataset[index], batch_size=1,
                      shuffle=False,
                      sampler=None, num_workers=2,
                      pin_memory=False)


