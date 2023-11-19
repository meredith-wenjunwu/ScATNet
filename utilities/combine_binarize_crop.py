import argparse
import os
import torch
import pdb

def main():
    parser = argparse.ArgumentParser(description='combine binarized dataset to form multi resolution dataset with different number of crops')

    parser.add_argument('--dataset_directories', required=True, type=str, nargs="+", default='results', help='Checkpoint directory location.')
    parser.add_argument('--scale_inds', required=True, type=int, nargs="+", help='indices of each dataset to combine')
    parser.add_argument('--write_directory', default='./', help='output directory for result pt files and experiment files')
    args = parser.parse_args()

    dataset_directories = args.dataset_directories
    scale_inds = args.scale_inds
    write_directory = args.write_directory
    if not os.path.exists(write_directory):
        os.makedirs(write_directory)

    assert len(dataset_directories) == len(scale_inds), "directories (len: {}) and indices (len: {}) provided does not match: ".format(len(dataset_directories),
                                                                                                                                       len(scale_inds))
    first_dir = dataset_directories[0]
    first_ind = scale_inds[0]
    for mode in ['train', 'valid', 'test']:
        experiment_txt = os.path.join(first_dir, mode+'.txt')
        assert os.path.exists(experiment_txt), "dataset file " + experiment_txt + "does not exist."
        with open(experiment_txt, 'r') as f:
            lines = [line for line in f]
        for line in lines:
            splitted = line.split(';')
            curr_file = splitted[0]
            to_save = []
            curr_file_feat = torch.load(curr_file)
            to_save.append(curr_file_feat[first_ind])
            # read in the same file from other directories
            for i in range(1, len(dataset_directories)):
                dataset_dir = dataset_directories[i]
                bn = os.path.basename(curr_file)
                other_file = os.path.join(dataset_dir, bn)
                other_file_feat = torch.load(other_file)
                other_ind = scale_inds[i]
                to_save.append(other_file_feat[other_ind])
            # save the feat
            out_experiment_txt = os.path.join(write_directory, mode+'.txt')
            out_feat_filename = os.path.join(write_directory, bn)
            torch.save(to_save, out_feat_filename)
            with open(out_experiment_txt, 'a') as f:
                to_write = out_feat_filename
                for l in splitted[1:]:
                    to_write += ';' + l
                f.write(to_write)


if __name__ == '__main__':
    main()