import os
from torch.utils.tensorboard import SummaryWriter


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
        self.writer = SummaryWriter(log_dir=os.path.join(self.directory))

    def add_scalar(self, *args):
        self.writer.add_scalar(*args)