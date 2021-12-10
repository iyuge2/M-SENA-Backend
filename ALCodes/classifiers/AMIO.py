"""
AMIO -- All Model in One
"""
import os
import sys
import torch
import torch.nn as nn


from classifiers.AlignNets import AlignSubNet
from classifiers.TFN import TFN

__all__ = ['AMIO']

MODEL_MAP = {
    'TFN': TFN,
}

class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.need_align = args.aligned
        text_seq_len, _, _ = args.seq_lens
        # simulating word-align network (for seq_len_T == seq_len_A == seq_len_V)
        if(self.need_align):
            self.alignNet = AlignSubNet(args, 'avg_pool')
            args.seq_lens = self.alignNet.get_seq_len()
        lastModel = MODEL_MAP[args.classifier.upper()]
        self.Model = lastModel(args)

    def forward(self, text_x, audio_x, video_x):
        if(self.need_align):
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x)
        return self.Model(text_x, audio_x, video_x)