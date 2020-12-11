"""
ASIO -- All Selector in One
"""
import torch
import torch.nn as nn

from discriminators.DEMO import DEMO 

__all__ = ['ASIO']

class ASIO():
    def __init__(self):
        self.SELECTOR_MAP = {
            'DEMO': DEMO,
        }

    def getSelector(self, args):
        return self.SELECTOR_MAP[args.selector.upper()](args)