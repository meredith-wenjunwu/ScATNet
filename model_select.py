
import glob
import os
import json
from main import main as main_op
from utilities.util import load_arguments
import argparse
import pdb
import collections
import torch
import shutil


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
        averaged_params[k].div_(num_models)
    return averaged_params


def generate_average_checkpts(epoch_numbers, checkpoint_dir, after_ep):
    # print(f'epochs to average: {epoch_numbers}')

    file_names = glob.glob('{}/*_*.pt'.format(checkpoint_dir))
    model_fnames = []

    for average_i in range(2, len(epoch_numbers) + 1):
        checkpoints = []
        curr_epoch_num = epoch_numbers[:average_i]
        for f_name in file_names:
            # first split on _ep_
            # then split on '.pth'
            if 'best' in f_name:
                continue

            ep_no = int(f_name.split('_')[-2].split('.pt')[0])
            if ep_no in curr_epoch_num:

                if not os.path.isfile(f_name):
                    print('File does not exist. {}'.format(f_name))
                else:
                    checkpoints.append(f_name)

        assert len(checkpoints) > 1, 'Atleast two checkpoints are required for averaging'
        averaged_weights = average_checkpoints(checkpoints)
        ckp_name = '{}/averaged_model_best{}.pt'.format(checkpoint_dir, average_i) if after_ep == -1 else \
            '{}/averaged_model_best{}_after{}.pt'.format(checkpoint_dir, average_i, after_ep)
        torch.save(averaged_weights, ckp_name)
        model_fnames.append(ckp_name)
        print('Finished writing averaged checkpoint')
    return model_fnames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select best checkpoints')

    parser.add_argument('--checkpoint-dir', required=True, type=str, default='results', help='Checkpoint directory location.')
    parser.add_argument('--multiple-dir', action='store_true', default=False)
    parser.add_argument('--load-config', action='store_true')
    parser.add_argument('--mode', default='train', choices=['test', 'valid', 'test-on-train-valid', 'test-on-train'])
    parser.add_argument('--after-ep', type=int, default=-1, help='Ignore first k epoch')
    parser.add_argument('--file-ext', type=str, default='pth', required=False)
    args = parser.parse_args()
    if args.multiple_dir:
        checkpoint_dirs = [os.path.join(args.checkpoint_dir, x) for x in os.listdir(args.checkpoint_dir) if os.path.isdir(os.path.join(args.checkpoint_dir, x))]
        checkpoint_dirs = sorted(checkpoint_dirs)
    else:
        checkpoint_dirs = [args.checkpoint_dir]
    for checkpoint_dir in checkpoint_dirs:

        config = glob.glob(os.path.join(checkpoint_dir, 'config*.json'))
        if args.load_config and len(config) > 0:
                args_all = load_arguments(parser, config[0])
                args_all.savedir = checkpoint_dir
        else:
            args_all = args
        curr_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*.' + args.file_ext)) + glob.glob(os.path.join(checkpoint_dir, '*.pt')))
        if len(glob.glob('{}/val_stats*'.format(checkpoint_dir))) >= 1:
            val_stats = json.load(open(glob.glob('{}/val_stats*'.format(checkpoint_dir))[0], 'r'))
            epoch = sorted(val_stats.keys(), key=int)
            epoch_acc = []
            for i, ep in enumerate(epoch):
                if int(ep) > args.after_ep:
                    epoch_acc.append((ep, min(val_stats[ep]["specificity_class"]) * val_stats[ep]['overall_accuracy'] * val_stats[ep]['true_negative_rate_class'][0] / (min(val_stats[ep]["specificity_class"]) + val_stats[ep]['overall_accuracy'] + val_stats[ep]['true_positive_rate_class'][0])))
            sorted_epoch_acc = sorted(epoch_acc, key=lambda x: (x[1], x[0]), reverse=True)
            epoch_numbers = [int(x[0]) for x in sorted_epoch_acc[:7]]
            model_fnames = generate_average_checkpts(epoch_numbers, checkpoint_dir, args.after_ep)
            for model_fname in model_fnames:
                args_all.resume = model_fname
                args_all.mode = args.mode
                result = main_op(vars(args_all))
                metric = min(result[2]["specificity_class"]) * result[2]['overall_accuracy'] * result[2]['true_negative_rate_class'][0] / (min(result[2]["specificity_class"]) + result[2]['overall_accuracy'] + result[2]['true_negative_rate_class'][0])  
                epoch_acc.append((model_fname, metric))
            sorted_epoch_acc = sorted(epoch_acc, key=lambda x: (x[1], x[0]), reverse=True)
            best_epoch = sorted_epoch_acc[0][0]
            pdb.set_trace()
            if 'average' in best_epoch:
                best_checkpt = best_epoch
            else:
                best_checkpt = glob.glob(os.path.join(checkpoint_dir, '*_{}_*.*'.format(best_epoch)))[0]

        else:
            result_summary = {}
            highest_metric= -1
            best_checkpt = None
            for checkpoint in curr_checkpoints:
                    args_all.resume = checkpoint
                    args_all.mode = args.mode
                    result = main_op(vars(args_all))
                    result_summary[os.path.basename(checkpoint)] = result[2]
                    metric = min(result[2]["specificity_class"]) * result[2]['overall_accuracy'] * result[2]['true_negative_rate_class'] / (min(result[2]["specificity_class"]) + result[2]['overall_accuracy'] + result[2]['true_negative_rate_class'])  
                    if (metric > highest_metric):
                        highest_metric = metric
                        best_checkpt = checkpoint
                        
            result_summary['best_checkpoint'] = best_checkpt
            output_fname = os.path.join(checkpoint_dir, 'val_stats.json')
            with open(output_fname, 'w') as json_file:
                json.dump(result_summary, json_file)
        
        # selected model
        args_all.resume = best_checkpt
        args_all.mode = 'test'
        args_all.save_result=True
        result = main_op(vars(args_all))
        print(json.dumps(result[2], indent=4, sort_keys=True))
        print('Best checkpoint for {} \n is {}'.format(os.path.basename(checkpoint_dir), os.path.basename(best_checkpt)))
        shutil.copyfile(best_checkpt, os.path.join(checkpoint_dir, 'best_model.{}'.format(os.path.splitext(best_checkpt)[1])))







