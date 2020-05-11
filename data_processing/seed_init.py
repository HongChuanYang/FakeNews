import torch
import random
import numpy as np


def init_seed():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
