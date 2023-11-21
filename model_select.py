
import glob
import os
import json
from main import main as main_op
from utilities.util import load_arguments
import argparse
import pdb


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
        result_summary = {}
        highest_metric= -1
        second_metric = -1
        best_checkpt = None
        config = glob.glob(os.path.join(checkpoint_dir, '*_train.json'))
        if args.load_config and len(config) > 0:
                args_all = load_arguments(parser, config[0])
        curr_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*.' + args.file_ext)) + glob.glob(os.path.join(checkpoint_dir, '*.pt')))
        for checkpoint in curr_checkpoints:
            args_all.resume = checkpoint
            args_all.mode = args.mode
            result = main_op(vars(args_all))
            result_summary[os.path.basename(checkpoint)] = result[2]
            metric = min(result[2]["specificity_class"]) * result[2]['overall_accuracy'] 
            second = result[2]["younden_index_micro"]
            if (metric > highest_metric) or (metric == highest_metric and second > second_metric):
                # pdb.set_trace()
                highest_metric = metric
                best_checkpt = checkpoint
                second_metric = second
        
        output_fname = os.path.join(checkpoint_dir, 'train_valid.json')
        with open(output_fname, 'w') as json_file:
            json.dump(result_summary, json_file)

        # selected model
        args_all.resume = best_checkpt
        args_all.mode = 'test'
        args_all.save_result=True
        result = main_op(vars(args_all))
        print(json.dumps(result[2], indent=4, sort_keys=True))







