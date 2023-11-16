import argparse
import collections
import torch
import glob
import os
import json
import shutil
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pdb
from main import main as main_op


def confusion_matrix_metrics(summary):
    if type(summary) == dict:
        # compute over prediction rate
        cmat = confusion_matrix(summary['true_labels'], summary['pred_labels'])
        cmat_np = np.array(cmat)
        total_sample = np.sum(cmat_np)
        under_prediction = 0
        for i in range(cmat_np.shape[0]):
            under_prediction += np.sum(cmat_np[i, :i+1])
        under_prediction /= total_sample
        score = summary['overall_accuracy'] * under_prediction / \
                (summary['overall_accuracy'] + under_prediction)
    else:
        train_cmat = confusion_matrix(summary[1]['true_labels'], summary[1]['pred_labels'])
        train_cmat_np = np.array(train_cmat)
        train_total_sample = np.sum(train_cmat_np)
        train_under_prediction = 0
        for i in range(train_cmat_np.shape[1]):
            train_under_prediction += np.sum(train_cmat_np[i, :i+1])
        train_under_prediction /= train_total_sample
        train_score = summary[1]['overall_accuracy'] * train_under_prediction / \
                (summary[1]['overall_accuracy'] + train_under_prediction)
        val_cmat = confusion_matrix(summary[0]['true_labels'], summary[0]['pred_labels'])
        val_cmat_np = np.array(val_cmat)
        val_total_sample = np.sum(val_cmat_np)
        val_under_prediction = 0
        for i in range(val_cmat_np.shape[1]):
            val_under_prediction += np.sum(val_cmat_np[i, :i+1])
        val_under_prediction /= val_total_sample
        val_score = summary[0]['overall_accuracy'] * val_under_prediction / \
                (summary[0]['overall_accuracy'] + val_under_prediction)

        score = val_score
    return score


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
    # best_file = glob.glob('{}/best.pt'.format(checkpoint_dir))[0]
    model_fnames = []

    for average_i in range(2, len(epoch_numbers) + 1):
        checkpoints = []
        curr_epoch_num = epoch_numbers[:average_i]
        for f_name in file_names:
            # first split on _ep_
            # then split on '.pth'
            if 'model' not in f_name or 'best' in f_name:
                continue

            ep_no = int(f_name.split('_')[-2].split('.pt')[0])
            if ep_no in curr_epoch_num:

                if not os.path.isfile(f_name):
                    print('File does not exist. {}'.format(f_name))
                else:
                    checkpoints.append(f_name)

        assert len(checkpoints) > 1, 'Atleast two checkpoints are required for averaging'
        averaged_weights = average_checkpoints(checkpoints)
        ckp_name = '{}/averaged_model_best{}.pth'.format(checkpoint_dir, average_i) if after_ep == -1 else \
            '{}/averaged_model_best{}_after{}.pth'.format(checkpoint_dir, average_i, after_ep)
        torch.save(averaged_weights, ckp_name)
        model_fnames.append(ckp_name)
        print('Finished writing averaged checkpoint')
    return model_fnames


def compare_test_checkpoints(config_name, model_fnames, checkpoint_dir):
    with open(config_name, 'r') as fp:
        opts = json.load(fp)
    # opts = Namespace(**json_dict)
    opts['mode'] = "valid"
    opts['visualize_specific_case'] = None
    opts['seed'] = 0
    opts['with_age'] = False
    scores = []
    for model_fname in model_fnames:
        opts['mode'] = "valid"
        opts['resume'] = model_fname
        opts['model_dir'] = checkpoint_dir
        opts['savedir'] = checkpoint_dir
        loss, val_acc, results_summary = main_op(opts)
        score = confusion_matrix_metrics(results_summary)
        opts['mode'] = "valid-train"
        _, train_acc, results_summary = main_op(opts)
        train_score = confusion_matrix_metrics(results_summary)
        scores.append(train_score + 1.5 * score)
    model_fname = model_fnames[np.argmax(scores)]
    opts['resume'] = model_fname
    opts['save_result'] = True
    opts['model_dir'] = os.path.dirname(model_fname)
    opts['savedir'] = os.path.dirname(model_fname)
    opts['mode'] = "test"
    opts['save_result'] = True
    main_op(opts)
    # import pdb;pdb.set_trace()
    os.rename(model_fname, os.path.join(checkpoint_dir, 'best_model.pth'))
    return {os.path.join(checkpoint_dir, 'best_model.pth'): np.max(scores)}


def main():
    parser = argparse.ArgumentParser(description='Select best checkpoints')

    parser.add_argument('--checkpoint-dir', required=True, type=str, default='results', help='Checkpoint directory location.')
    parser.add_argument('--multiple-dir', action='store_true', default=False)
    parser.add_argument('--load-config', required=True, type=str,
                        default='/projects/patho2/melanoma_diagnosis/model/2scale_50/config_cropsize_6144x12288_class_7.5_12.5_multi_resolution_train.json')
    parser.add_argument('--after-ep', type=int, default=-1, help='Ignore first k epoch')

    args = parser.parse_args()
    if args.multiple_dir:
        checkpoint_dirs = [os.path.join(args.checkpoint_dir, x) for x in os.listdir(args.checkpoint_dir) if os.path.isdir(os.path.join(args.checkpoint_dir, x))]
    else:
        checkpoint_dirs = [args.checkpoint_dir]
    best_model_dict = {}
    for checkpoint_dir in checkpoint_dirs:
        # if 'best_model.pth' in os.listdir(checkpoint_dir):
        #     continue
        if len(glob.glob('{}/val_stats*'.format(checkpoint_dir))) < 1:
            continue

        summary = json.load(open(glob.glob('{}/val_stats*'.format(checkpoint_dir))[0], 'r'))
        epoch_acc = [(int(ep), confusion_matrix_metrics(summary[str(ep)])) for
        ep in range(200) if (ep < 80) and (ep > args.after_ep)]
        # sort according to confusion matrix
        sorted_epoch_acc = sorted(epoch_acc, key=lambda x: (x[1], x[0]), reverse=True)
        epoch_numbers = [x[0] for x in sorted_epoch_acc[:6]]
        model_fnames = generate_average_checkpts(epoch_numbers, checkpoint_dir, args.after_ep)
        best_model_dict.update(compare_test_checkpoints(args.load_config, model_fnames, checkpoint_dir))
    if args.multiple_dir:
        scores_multiple_dirs = best_model_dict.values()
        best_best_model_path = list(best_model_dict.keys())[np.argmax(list(scores_multiple_dirs))]
        shutil.copyfile(best_best_model_path, os.path.join(args.checkpoint_dir, 'best_model.pth'))    



if __name__ == '__main__':
    main()
