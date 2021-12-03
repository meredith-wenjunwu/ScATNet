import argparse
import collections
import torch
import glob
import os
import json
import pdb


def average_checkpoints(model_checkpoints):
    params_dict = collections.OrderedDict()
    params_keys = None
    new_model_params = None
    num_models = len(model_checkpoints)

    for ckpt in model_checkpoints:
        model_params = torch.load(ckpt, map_location='cpu')

        # Copies over the settings from the first checkpoint
        if new_model_params is None:
            new_model_params = model_params

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(ckpt, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:

                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        averaged_params[k] = averaged_params[k].div(num_models)
    return averaged_params


def main():
    parser = argparse.ArgumentParser(description='Average checkpoints')

    parser.add_argument('--checkpoint-dir', required=True, type=str, default='results', help='Checkpoint directory location.')
    parser.add_argument('--best-n', required=True, type=int, default=5, help='Num of epochs to average')
    parser.add_argument('--after_epoch', required=False, type=int, default=0, help='Average best after specific epochs')
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    '''
    epoch_acc = json.load(open(glob.glob('{}/val*.json'.format(checkpoint_dir))[0], 'r'))
    epoch_acc = [(int(ep), acc) for ep, acc in epoch_acc.items() if int(ep) > args.after_ep]
    sorted_epoch_acc = sorted(epoch_acc, key=lambda x: (x[1], x[0]), reverse=True)

    epoch_numbers = [x[0] for x in sorted_epoch_acc[:args.best_n]]
    print(f'epochs to average: {epoch_numbers}')
    '''
    file_names = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
    after_epochs_name = []
    if args.after_epoch > 0:
        for f_name in file_names:
            epoch = int(os.path.splitext(os.path.basename(f_name))[0].split('_')[-2])
            if epoch >= args.after_epoch:
                after_epochs_name.append(f_name)
    else:
        after_epochs_name = file_names
    after_epochs_name = set(after_epochs_name) - set(glob.glob(os.path.join(checkpoint_dir, '*EMA*.pt')))
    epoch_acc = [(int(os.path.splitext(os.path.basename(x))[0].split('_')[-2]),
                  float(os.path.splitext(os.path.basename(x))[0].split('_')[-1])) for x in after_epochs_name]
    sorted_epoch_acc = sorted(epoch_acc, key=lambda x: (x[1], x[0]), reverse=True)
    epoch_numbers = [x[0] for x in sorted_epoch_acc[:args.best_n]]

    checkpoints = []

    for f_name in after_epochs_name:
        if 'EMA' in f_name:
            continue
        # first split on _ep_
        # then split on '.pth'
        ep_no = int(f_name.split('_')[-2].split('.pt')[0])
        if ep_no in epoch_numbers:
            if not os.path.isfile(f_name):
                print('File does not exist. {}'.format(f_name))
            else:
                checkpoints.append(f_name)

    assert len(checkpoints) > 1, 'Atleast two checkpoints are required for averaging'
    print(checkpoints)

    averaged_weights = average_checkpoints(checkpoints)
    ckp_name = '{}/averaged_model_best{}.pth'.format(checkpoint_dir, args.best_n)

    torch.save(averaged_weights, ckp_name)

    print('Finished writing averaged checkpoint after epoch {}'.format(args.after_epoch))


if __name__ == '__main__':
    main()
