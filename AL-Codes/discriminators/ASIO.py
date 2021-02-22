"""
ASIO -- All Selector in One
"""
import torch
import torch.nn as nn

from discriminators.CONF import CONF
from discriminators.MARGIN import MARGIN
from discriminators.ENTROPY import ENTROPY

__all__ = ['ASIO']

class ASIO():
    def __init__(self):
        self.SELECTOR_MAP = {
            'CONF': CONF,
            'MARGIN': MARGIN,
            'ENTROPY': ENTROPY,
        }

    def getSelector(self, args):
        return self.SELECTOR_MAP[args.selector.upper()](args)