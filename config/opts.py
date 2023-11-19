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
    group.add_argument('--load-config', default=None, type=str, help='path to config')
    group.add_argument('--mode', default='train', choices=['train', 'test', 'valid', 'test-on-train-valid', 'test-on-train' 'merge-train-valid'],
                        help='Experiment mode')
    group.add_argument('--resize1-scale', default=[1.0], type=float, nargs="+", help='number of scales for image')
    group.add_argument('--resize2-scale', default=[1.0], type=float, nargs="+", help='number of scales for crops')
    group.add_argument('--model', default='model_v2', choices=supported_models)
    group.add_argument('--seed', default=1669, type=int,
                        help='random seed for pytorch')
    group.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    group.add_argument('--batch-size', default=5, type=int,
                        help='batch size (default: 5)')
    parser.add_argument('--epochs', default=100,
                        type=int, help='number of maximum epochs to run')
    parser.add_argument('--start-epoch', default=0,
                        type=int, help='manual epoch number (useful on restarts)')
    group.add_argument('--finetune', action='store_true', default=False, help='Freeze batch norm layer for fine tuning')
    group.add_argument('--finetune-base-extractor', action='store_true', default=False, help="Freeze batch norm layer in base extractor for fine tuning")
    group.add_argument('--mixed_precision', action='store_true', default=False, help='Training with mixed precision')
    group.add_argument('--max-bsz-cnn-gpu0', default=100, type=int, help='Max. batch size on GPU0')
    group.add_argument('--resume', default=None, type=str, help='path to latest checkpoint (default: none)')
    group.add_argument('--use-gpu', action='store_true', default=False, help='Use gpu for experiment')
    group.add_argument('--gpu-id', nargs='+', type=int)
    group.add_argument('--use-parallel', action='store_true', default=False, help='whether to use data parallel or not')
    group.add_argument('--warmup', action='store_true', default=False, help='whether to warm up or not')
    '''
    stage_dict = {'123v45.pt': {0: '12v3.pt', 1: '4v5.pt'},
                  '12v3.pt': {0: '1v2.pt', 1: 3},
                  '4v5.pt': {0: 4, 1: 5},
                  '1v2.pt': {0: 1, 1: 2}}
    '''
    return parser


def binarize_opts(parser):
    group = parser.add_argument_group('Binarize Options')
    group.add_argument('--binarize', action='store_true', default=False)
    group.add_argument('--num-crops', type=int, nargs='+', default=[7])
    return parser


def visualization_opts(parser):
    group = parser.add_argument_group('Visualization options')
    group.add_argument('--visdom', action='store_true', default=False,
                        help='Track training progress using visdom')
    group.add_argument('--visdom-port', default=8097, help='Port to plot visdom')
    group.add_argument('--savedir', type=str, default='./',
                        help='Location to save the results')
    group.add_argument('--save-result', action='store_true', default=False,
                        help='save results to txt files or not')
    group.add_argument('--save-top-k', action='store_true', default=False,
                       help='Saving top k crops')
    return parser



def get_dataset_opts(parser):
    '''
        Medical imaging Dataset details
    '''
    group = parser.add_argument_group('Dataset general details')
    group.add_argument('--data', '--datasetfile', help='path to dataset txt files')
    group.add_argument('--attn', help='path to attention guidance weights')
    group.add_argument('--evaluate-by-case', action='store_true', default=False, help='Is there multiple slice per case?')
    group.add_argument('--mask', default=None,
                        help='path to masks (background, dermis, epidermis)')
    group.add_argument('--mask-type', default=None, choices=['black-bg',
                                                             'return-indices'],
                        help='Types of mask to apply to input')
    group.add_argument('--num-classes', default=5, type=int, help='Number of classes')
    group.add_argument('--dataset-crop', default=[1,0, 1.0], help='ratio of crop to drop in training', nargs='+',
                       type=float)
    group.add_argument('--resize1', '--crop1', type=int, nargs='+')
    group.add_argument('--resize2', '--crop2', default=[512, 512], type=int, nargs='+')
    group.add_argument('--transform', type=str, choices=['Zooming', 'DivideToScale'], default='DivideToScale',
                       help='Type of transform: 1) Zooming - random zoom and center crop, then divide to scale' + \
                       '; 2) DivideToScale - just divide original image to different scales')
    group.add_argument('--mask-threshold', default=0.4, type=float,
                       help='percentage of background for crops to be removed')
    return parser

def get_model_opts(parser):
    '''Model details'''
    group = parser.add_argument_group('Medical Imaging Model Details')
    # group.add_argument('--s', default=1.0, type=float, help='Factor by which output channels should be scaled (s > 1 '
                                                             # 'for increasing the dims while < 1 for decreasing)')
    group.add_argument('--model-dir', type=str,
                        default='./model/test.pth',
                        help='directory to output saved checkpoint')
    group.add_argument('--channels', default=3, type=int, help='Input channels')
    group.add_argument('--model-dim', default=256, type=int, help="linear projection dimension")
    group.add_argument('--weight-tie', action='store_true', default=False, help='use weight tying for transformers')
    group.add_argument('--attn_guide', action='store_true', default=False, help='use attention guiding regularization')
    group.add_argument('--attn_head', default=2, type=int, help='number of heads to impose attention guidance')
    group.add_argument('--n-layers', default=4, type=int, help='number of attention layers')
    group.add_argument('--head-dim', default=64, type=int, help="head dimension for attention layers")
    group.add_argument('--drop-out', default=0.2, type=float)
    group.add_argument('--linear-channel', default=4, type=int, help="feed forward dimension")
    group.add_argument('--self-attention', action='store_true', default=False)
    group.add_argument('--variational-dropout', action='store_true', default=False, help='Use variational dropout')
    group.add_argument('--in-dim', default=1280, type=int, help='output feature dimension from feature extractor')
    group.add_argument('--use-standard-emb', action='store_true', default=False)
    group.add_argument('--num-scale-attn-layer', type=int, default=2, help='number of attention layers for scale aware')
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
    group.add_argument('--weight-decay', default=4e-5, type=float, help='Weight decay')
    group.add_argument('--lambda-attn', default=0.1, type=float, help='weight for attention loss')
    group.add_argument('--attn-loss', choices=["Frobenius", "Inclusion_Exclusion"], type=str, help='attention loss type')
    group = parser.add_argument_group('Optimizer accumulation options')
    group.add_argument('--aggregate-batch', default=1, type=int, help="aggregate gradient for number of batches")
    return parser

def get_scheduler_opts(parser):
    ''' Scheduler Details'''
    group = parser.add_argument_group('Scheduler Opts')
    group.add_argument('--patience',
                        default=50, type=int,
                        help='Max. epochs to continue running for scheduler')
    group.add_argument('--scheduler',
                        default='step', choices=['step', 'cosine', 'reduce', 'cycle'], type=str,
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
    if args.mode == 'train':
        model_dir = '{}_{}x{}_{}'.format(os.path.basename(os.path.dirname(args.data)),
                                         args.resize1[0], args.resize1[1],
                                         timestr)
    else:
        model_dir = ''
    if args.model_dir is None:
        args.model_dir = model_dir
    else:
        args.model_dir = os.path.join(args.model_dir, model_dir)
    if args.savedir is None or args.savedir == '':
        args.savedir = args.model_dir
    else:
        args.savedir = os.path.join(args.savedir, model_dir)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)


    return args, parser
