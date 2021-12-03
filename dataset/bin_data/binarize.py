import torch
from os import path
from config.build import build_dataset, build_model, build_cuda
import tqdm
import os
import pdb


def multi_scale_features(multi_data, feature_extractor):
    multi_feat = []
    num_scales = len(multi_data)
    n_gpus = torch.cuda.device_count()
    dev = torch.device('cuda') if n_gpus > 0 else torch.device('cpu')
    feature_extractor.eval()
    with torch.no_grad():
        for j in range(num_scales):
            _data = multi_data[j]
            if feature_extractor is not None:
                split_data = torch.split(_data, split_size_or_sections=1, dim=1)
                feat = []
                for data in split_data:
                    bsz, n_split, channels, width, height = data.size()
                    assert bsz == 1
                    try:
                        data = data.to(device=dev)
                        data = data.contiguous().view(n_split, channels, width, height)
                        data = feature_extractor(data)
                    except RuntimeError as e:
                        data = data.cpu()
                        feature_extractor = feature_extractor.cpu()
                        data = feature_extractor(data)
                    # reshape
                    feature_extractor = feature_extractor.to(device=dev)
                    data = data.contiguous().view(n_split, -1)
                    feat.append(data)
                feat = torch.cat(feat, dim=0)
                feat = feat.cpu()
                multi_feat.append(feat)
    return multi_feat


def binarize_data(data_loader, feature_extractor, output_path='', name='train'):
        for i, (multi_data, labels, labels_conf, paths, mask) in tqdm.tqdm(enumerate(data_loader)):
            img_name = paths[0].replace('.tif', '.pt')
            img_name = os.path.join(output_path, os.path.basename(img_name))
            if os.path.exists(img_name):
                continue
            features = multi_scale_features(multi_data=multi_data, feature_extractor=feature_extractor)
            img_name = paths[0].replace('.tif', '.pt')
            img_name = os.path.join(output_path, os.path.basename(img_name))
            torch.save(features, img_name)
            labels_conf_str = ''
            for i in labels_conf.cpu().tolist()[0]:
                labels_conf_str += str(i) + ','
            labels_conf_str = labels_conf_str[:-1]
            with open(os.path.join(output_path, '{}.txt'.format(name)), 'a') as f:
                f.write('{};{};{}\n'.format(img_name, labels.cpu().item(), labels_conf_str))


def main_binarize(opts):
    opts['batch_size'] = 1
    opts['binarize'] = True

    train_loader, valid_loader, test_loader = build_dataset(opts)
    opts = build_cuda(opts)
    model, feature_extractor = build_model(opts)
    # need to write experiment_txt file
    print('Processing training data')
    binarize_data(data_loader=train_loader, feature_extractor=feature_extractor, output_path=opts['savedir'],
                   name='train')
    print('Processing validation data')
    binarize_data(data_loader=valid_loader, feature_extractor=feature_extractor, output_path=opts['savedir'],
                  name='valid')

    print('Processing test data')
    binarize_data(data_loader=test_loader, feature_extractor=feature_extractor, output_path=opts['savedir'],
                  name='test')
