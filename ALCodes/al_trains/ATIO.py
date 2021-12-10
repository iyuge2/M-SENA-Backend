"""
AIO -- All Trains in One
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal

from al_trains.TFN import TFN

__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            # single-task
            'TFN': TFN,
        }
    
    def getTrain(self, args):
        return self.TRAIN_MAP[args.classifier.upper()](args)