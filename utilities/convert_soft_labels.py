import argparse
import os
import torch
import pdb
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='combine binarized dataset to form multi resolution dataset with different number of crops')

    parser.add_argument('--dataset_directories', required=True, type=str, default='results', help='Checkpoint directory location.')
    parser.add_argument('--write_directory', default='./', help='output directory for result pt files and experiment files')
    args = parser.parse_args()

    dataset_directories = args.dataset_directories
    write_directory = args.write_directory
    if not os.path.exists(write_directory):
        os.makedirs(write_directory)
    for mode in ['train', 'valid', 'test']:
        experiment_txt = os.path.join(dataset_directories, mode + '.txt')
        assert os.path.exists(experiment_txt), "dataset file " + experiment_txt + "does not exist."
        with open(experiment_txt, 'r') as f:
            lines = [line.rstrip() for line in f]
        for line in lines:
            im_p, label, select = line.split(';')
            label = int(label)
            select = [float(s) for s in select.split(',')]
            if len(np.unique(select)) == 2 or mode in ['valid', 'test']:
                assert 0 in select, "{}".format(select)
                select = [str(int(s)) for s in select]
                uniform_conf = ','.join(select)
                constraint_conf = uniform_conf
            else:
                uniform_conf  = "0.25, 0.25, 0.25, 0.25"
                if label == 0:
                    constraint_conf = "1, 0, 0, 0"
                else:
                    each = 1.0/(label + 1)
                    constraint_conf = str(each)
                    for i in range(1, 4):
                        if i <= label:
                            constraint_conf += ', '+ str(each)
                        else:
                            constraint_conf += ', 0'


            # write uniform and uniform constraint
            # read in the same file from other directories
            uniform_out_dir = os.path.join(write_directory, 'uniform')
            uniform_constraint_out_dir = os.path.join(write_directory, 'uniform_constraint')
            if not os.path.exists(uniform_out_dir):
                os.makedirs(uniform_out_dir)
            if not os.path.exists(uniform_constraint_out_dir):
                os.makedirs(uniform_constraint_out_dir)
            uniform_out_txt = os.path.join(uniform_out_dir, mode+'.txt')
            uniform_out_constraint_txt = os.path.join(uniform_constraint_out_dir, mode + '.txt')
            with open(uniform_out_txt, 'a') as f:
                f.write('{};{};{}\n'.format(im_p, label, uniform_conf))
            with open(uniform_out_constraint_txt, 'a') as f:
                f.write('{};{};{}\n'.format(im_p, label, constraint_conf))


if __name__ == '__main__':
    main()