import os
import torch
import argparse
import json
import glob
from utilities.save_dict_to_file import DictWriter
from utilities.print_utilities import *


def save_arguments(args, save_loc, json_file_name='arguments.json'):
    argparse_dict = vars(args)
    arg_fname = '{}/{}'.format(save_loc, json_file_name)
    writer = DictWriter(file_name=arg_fname, format='json')
    writer.write(argparse_dict)
    print_log_message('Arguments are dumped here: {}'.format(arg_fname))


def load_arguments(parser, dumped_arg_loc):
    arg_fname = dumped_arg_loc
    parser = argparse.ArgumentParser(parents=[parser], add_help=False)
    with open(arg_fname, 'r') as fp:
        json_dict = json.load(fp)
        parser.set_defaults(**json_dict)

        updated_args = parser.parse_args()

    return updated_args


def save_metrics(metrics, save_loc):
    writer = DictWriter(file_name=save_loc, format='json')
    writer.write(metrics)
    print_log_message('Results metrics dumped here: {}'.format(save_loc))