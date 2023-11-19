import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import datetime
import pdb
import torch
from os.path import join
import pandas as pd


class Logger:
    def __init__(self, configs):
        self.writer = SummaryWriter(join('/projects/patho2/melanoma_diagnosis/runs', datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]") +
                                         configs['save_name']))
        self.num_classes = configs['num_classes']



    def update(self, n_iter, value, mode=None):
        assert mode in ['train_loss', 'train_err', 'test_loss', 'test_err',
                       'val_loss', 'val_err', 'train_mat',
                        'val_mat', 'test_mat', 'lr', 'ema_loss',
                        'ema_mat', 'ema_err'], "Invalid Type, got {}".format(mode)
        if mode == 'train_loss':
            self.writer.add_scalar('Loss/train', value, n_iter)
        elif mode == 'train_err':
            self.writer.add_scalar('Accuracy/train', value, n_iter)
        elif mode == 'val_loss':
            self.writer.add_scalar('Loss/valid', value, n_iter)
        elif mode == 'val_err':
            self.writer.add_scalar('Accuracy/valid', value, n_iter)
        elif mode == 'ema_err':
            self.writer.add_scalar('Accuracy/ema', value, n_iter)
        elif mode == 'ema_loss':
            self.writer.add_scalar('Loss/ema', value, n_iter)
        elif mode == 'ema_mat':
            assert len(value) >= 2, "Invalid length"
            fig = plt.figure()
            value = pd.DataFrame(value, range(self.num_classes), range(self.num_classes))
            sns.heatmap(value, annot=True)
            self.writer.add_figure('Confusion Matrix/ema', fig, global_step=n_iter)
        elif mode == 'test_loss':
            self.writer.add_scalar('Loss/test', value, n_iter)
        elif mode == 'test_err':
            self.writer.add_scalar('Accuracy/test', value, n_iter)
        elif mode == 'train_mat':
            # value is conf_matrix
            fig = plt.figure()
            value = pd.DataFrame(value, range(self.num_classes), range(self.num_classes))
            sns.heatmap(value, annot=True)
            self.writer.add_figure('Confusion Matrix/train', fig, global_step=n_iter)
        elif mode == 'val_mat':
            assert len(value) >= 2, "Invalid length"
            fig = plt.figure()
            value = pd.DataFrame(value, range(self.num_classes), range(self.num_classes))
            sns.heatmap(value, annot=True)
            self.writer.add_figure('Confusion Matrix/valid', fig, global_step=n_iter)
        elif mode == 'test_mat':
            fig = plt.figure()
            value = pd.DataFrame(value, range(self.num_classes), range(self.num_classes))
            sns.heatmap(value, annot=True)
            self.writer.add_figure('Confusion Matrix/test', fig, global_step=n_iter)
        elif mode == 'lr':
            self.writer.add_scalar('LR', value, n_iter)
        else:
            "Invalid Type"
    def close(self):
        self.writer.close()
