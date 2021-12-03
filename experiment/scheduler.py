import math
import argparse

class PolynomialScheduler(object):
    def __init__(self, opts, **kwargs):
        is_iter_based = getattr(opts, "scheduler.is_iteration_based", False)
        max_iterations = getattr(opts, "scheduler.max_iterations", 50000)
        warmup_iterations = getattr(opts, "scheduler.warmup_iterations", 0)
        max_epochs = getattr(opts, "scheduler.max_epochs", 350)
        super(PolynomialScheduler, self).__init__(opts=opts)
        self.start_lr = getattr(opts, "scheduler.poly.start_lr", 0.1)
        self.end_lr = getattr(opts, "scheduler.poly.end_lr", 1e-5)
        self.power = getattr(opts, "scheduler.poly.power", 2.5)
        self.warmup_iterations = max(warmup_iterations, 0)
        if self.warmup_iterations > 0:
            warmup_init_lr = getattr(opts, "scheduler.warmup_init_lr", 1e-7)
            self.warmup_init_lr = warmup_init_lr
            self.warmup_step = (self.start_lr - self.warmup_init_lr) / self.warmup_iterations
        self.is_iter_based = is_iter_based
        self.max_iterations = max_iterations
        self.max_epochs = max_epochs
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Polynomial LR arguments", description="Polynomial LR arguments")
        group.add_argument('--scheduler.poly.power', type=float, default=2.5, help="Polynomial power")
        group.add_argument('--scheduler.poly.start-lr', type=float, default=0.1, help="Start LR in Poly LR scheduler")
        group.add_argument('--scheduler.poly.end-lr', type=float, default=1e-5, help="End LR in Poly LR scheduler")
        return parser
    def get_lr(self, epoch, curr_iter):
        if curr_iter < self.warmup_iterations:
            curr_lr = self.warmup_init_lr + curr_iter * self.warmup_step
        else:
            curr_iter = curr_iter - self.warmup_iterations
            factor = (curr_iter / self.max_iterations) if self.is_iter_based else (epoch / self.max_epochs)
            curr_lr = (self.start_lr - self.end_lr) * ((1.0 - factor) ** self.power) + self.end_lr
        return curr_lr

class CosineLR(object):
    def __init__(self, base_lr, max_epochs, optimizer):
        super(CosineLR, self).__init__()
        self.base_lr = base_lr
        self.max_epochs = max_epochs
        self.optimizer = optimizer

    def step(self, epoch):

        for pg in self.optimizer.param_groups:
            pg['lr'] = round(self.base_lr * (1 + math.cos(math.pi * epoch / self.max_epochs)) / 2, 6)

    def __repr__(self):
        fmt_str = 'Scheduler ' + self.__class__.__name__ + '\n'
        fmt_str += '    Total Epochs: {}\n'.format(self.max_epochs)
        fmt_str += '    Base LR : {}\n'.format(self.base_lr)
        return fmt_str
