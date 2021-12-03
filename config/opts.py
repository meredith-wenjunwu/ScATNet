import argparse
import random
import torch
import time
from model.base_feature_extractor import get_base_extractor_opts
from config import supported_optimziers, supported_models
from utilities.util import load_arguments
import os
import json
'''
In this file, we define command-line arguments

'''

def size(arg):
    return [int(x) for x in arg.split(',')]

def general_opts(parser):
    group = parser.add_argument_group('General Options')
    group.add_argument('--load_config', default=None, type=str, help='path to config')
    group.add_argument('--mode', default='train', choices=['train', 'test', 'valid', 'valid-train'],
                        help='Experiment mode')
    group.add_argument('--resize1_scale', default=[1.0], type=float, nargs="+", help='number of scales for image')
    group.add_argument('--resize2_scale', default=[1.0], type=float, nargs="+", help='number of scales for crops')
    group.add_argument('--model', default='model_v2', choices=supported_models)
    group.add_argument('--seed', default=1669, type=int,
                        help='random seed for pytorch')
    group.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    group.add_argument('--batch_size', default=5, type=int,
                        help='batch size (default: 5)')
    parser.add_argument('--epochs', default=100,
                        type=int, help='number of maximum epochs to run')
    parser.add_argument('--start_epoch', default=0,
                        type=int, help='manual epoch number (useful on restarts)')
    group.add_argument('--finetune', default=False, type=str_to_bool, help='Freeze batch norm layer for fine tuning')
    group.add_argument('--finetune_base_extractor', type=str_to_bool, help="Freeze batch norm layer in base extractor for fine tuning")
    group.add_argument('--mixed_precision', type=str_to_bool, default=False)
    group.add_argument('--eval_split', default=0, type=int, help='split data in half when evaluating to save memory space')
    group.add_argument('--train_split', default=0, type=int, help='split data in half when training to save memory space')
    group.add_argument('--resume', default=None, type=str, help='path to latest checkpoint (default: none)')
    group.add_argument('--use_gpu', default=True, type=str_to_bool, nargs='?', const=True, help='Use gpu for experiment')
    group.add_argument('--gpu_id', nargs='+', type=int)
    group.add_argument('--use_parallel', type=str_to_bool, help='whether to use data parallel or not')
    group.add_argument('--warmup', type=str_to_bool, default=False, help='whether to warm up or not')
    '''
    stage_dict = {'123v45.pt': {0: '12v3.pt', 1: '4v5.pt'},
                  '12v3.pt': {0: '1v2.pt', 1: 3},
                  '4v5.pt': {0: 4, 1: 5},
                  '1v2.pt': {0: 1, 1: 2}}
    '''
    return parser


def binarize_opts(parser):
    group = parser.add_argument_group('Binarize Options')
    group.add_argument('--binarize', default=False, type=str_to_bool)
    group.add_argument('--num_crops', type=int, default=7)
    return parser


def visualization_opts(parser):
    group = parser.add_argument_group('Visualization options')
    group.add_argument('--visdom', default=False, type=str_to_bool,
                        nargs='?', const=True, help='Track training progress using visdom')
    group.add_argument('--visdom_port', default=8097, help='Port to plot visdom')
    # group.add_argument('--plot_heatmaps', default=False, type=str_to_bool,
    #                     nargs='?', const=True, help='Plot and Save the heatmaps in intermediate layers')
    group.add_argument('--savedir', type=str, default='./',
                        help='Location to save the results')
    group.add_argument('--save_result', default=False, type=str_to_bool,
                        help='save results to txt files or not')
    group.add_argument('--save_top_k', default=-1, help='flag for saving top k crops',
                       type=int)
    return parser



def get_dataset_opts(parser):
    '''
        Medical imaging Dataset details
    '''
    group = parser.add_argument_group('Dataset general details')
    group.add_argument('--data', '--datasetfile', help='path to dataset txt files')
    group.add_argument('--dataset', choices=['melanoma', 'breakhis', 'ISIC'])
    group.add_argument('--mask', default=None,
                        help='path to masks (background, dermis, epidermis)')
    group.add_argument('--mask_type', default=None, choices=['black-bg',
                                                             'return-indices'],
                        help='Types of mask to apply to input')
    group.add_argument('--num_classes', default=5, type=int, help='Number of classes')
    group.add_argument('--dataset_crop', default=[1,0, 1.0], help='ratio of crop to drop in training', nargs='+',
                       type=float)
    group.add_argument('--resize1', '--crop1', type=int, nargs='+')
    group.add_argument('--resize2', '--crop2', default=256, type=int, nargs='+')
    group.add_argument('--transform', type=int, choices=[0, 1, 2, 3, 4], default=0)
    group.add_argument('--mask_threshold', default=0.4, type=float,
                       help='percentage of background for crops to be removed')
    group.add_argument('--sliding_window_evaluation', type=str_to_bool, default=False,
                       help='use sliding window for evaluation')
    return parser

def get_model_opts(parser):
    '''Model details'''
    group = parser.add_argument_group('Medical Imaging Model Details')
    # group.add_argument('--s', default=1.0, type=float, help='Factor by which output channels should be scaled (s > 1 '
                                                             # 'for increasing the dims while < 1 for decreasing)')
    group.add_argument('--model_dir', type=str,
                        default='/projects/patho1/melanoma_diagnosis/model/Shima/12v5sure_set1_bg_2048',
                        help='directory to output saved checkpoint')
    group.add_argument('--channels', default=3, type=int, help='Input channels')
    group.add_argument('--model_dim', default=256, type=int, help="linear projection dimension")
    group.add_argument('--weight-tie', default=True, type=str_to_bool, help='use weight tying for transformers')
    group.add_argument('--n_layers', default=4, type=int, help='number of attention layers')
    group.add_argument('--head_dim', default=64, type=int, help="head dimension for attention layers")
    group.add_argument('--drop_out', default=0.2, type=float)
    group.add_argument('--linear_channel', default=4, type=int, help="feed forward dimension")
    group.add_argument('--self_attention', default=False, type=str_to_bool)
    group.add_argument('--variational_dropout', default=False, type=str_to_bool, help='Use variational dropout')
    group.add_argument('--in_dim', default=1280, type=int, help='output feature dimension from feature extractor')
    group.add_argument('--use_standard_emb', type=str_to_bool, default=True)
    group.add_argument('--num_scale_attn_layer', type=int, default=2, help='number of attention layers for scale aware')
    return parser

def get_optimizer_opts(parser):
    'Loss function details'
    group = parser.add_argument_group('Optimizer options')
    group.add_argument('--smoothing', type=float, default=0.1,
                       help="smoothing for cross entropy")
    group.add_argument('--loss-function',
                       choices=['cross_entropy', 'smoothing', 'bce'],
                       type=str,
                       default='cross_entropy')
    group.add_argument('--optim', default='sgd', type=str, choices=supported_optimziers,
                       help='Optimizer')
    group.add_argument('--adam-beta1', default=0.9, type=float, help='Beta1 for ADAM')
    group.add_argument('--adam-beta2', default=0.999,  type=float, help='Beta2 for ADAM')
    group.add_argument('--weight-decay', default=4e-6, type=float, help='Weight decay')
    group = parser.add_argument_group('Optimizer accumulation options')
    group.add_argument('--aggregate_batch', default=1, type=int, help="aggregate gradient for number of batches")
    return parser

def get_scheduler_opts(parser):
    ''' Scheduler Details'''
    group = parser.add_argument_group('Scheduler Opts')
    group.add_argument('--patience',
                        default=50, type=int,
                        help='Max. epochs to continue running for scheduler')
    group.add_argument('--scheduler',
                        default=None, choices=['step', 'cosine', 'reduce', 'cycle'], type=str,
                        help='Learning rate scheduler')
    group.add_argument('--lr', default=0.0005, type=float,
                        help='initial learning rate')
    group.add_argument('--lr-decay', default=0.5, type=float,
                        help='factor by which lr should be decreased')
    return parser

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n', 'False'}:
        return False
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    #raise ValueError(f'{value} is not a valid boolean value')


def get_opts(parser):
    '''General options'''
    parser = general_opts(parser)

    '''Optimzier options'''
    parser = get_optimizer_opts(parser)

    '''Medical Image model options'''
    parser = get_model_opts(parser)

    '''Dataset related options'''
    parser = get_dataset_opts(parser)

    ''' LR scheduler details'''
    parser = get_scheduler_opts(parser)

    '''Base feature extractor options'''
    parser = get_base_extractor_opts(parser)


    return parser


def get_config():
    parser = argparse.ArgumentParser(description='Medical Imaging')
    parser = get_opts(parser)
    parser = visualization_opts(parser)
    parser = binarize_opts(parser)

    args = parser.parse_args()
    if args.load_config is not None:
        args = load_arguments(parser, args.load_config)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.set_num_threads(args.workers)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    if args.savedir is None:
        args.savedir = '{}_{}x{}_{}_{}'.format(args.mode,
                                               args.resize1[0][1], args.resize1[0][0],
                                               os.path.basename(os.path.dirname(args.data)),
                                               timestr)
    elif not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    return args, parser