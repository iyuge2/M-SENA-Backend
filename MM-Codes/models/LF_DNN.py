"""
paper1: Benchmarking Multimodal Sentiment Analysis
paper2: Recognizing Emotions in Video Using Multimodal DNN Feature Fusion
From: https://github.com/rhoposit/MultimodalDNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.subNets.FeatureNets import SubNet, TextSubNet

__all__ = ['LF_DNN']

class LF_DNN(nn.Module):
    """
    late fusion using DNN
    """
    def __init__(self, args):
        super(LF_DNN, self).__init__()
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden = args.hidden_dim_t
        self.audio_hidden = args.hidden_dim_a
        self.video_hidden = args.hidden_dim_v

        self.text_out= args.text_out
        self.post_fusion_dim = args.post_fusion_dim

        self.audio_prob = args.dropouts_a
        self.video_prob = args.dropouts_v
        self.text_prob = args.dropouts_t
        self.post_fusion_prob = args.dropouts_f

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(self.text_out + self.video_hidden + self.audio_hidden, self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, args.num_classes)


    def forward(self, text_x, audio_x, video_x):
        """
        text_x: (batch_size, seq_len, feature_len)
        audio_x: (batch_size, 1, feature_len)
        video_x: (batch_size, 1, feature_len)
        """
        audio_x = audio_x.squeeze(1)
        video_x = video_x.squeeze(1)

        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)

        fusion_h = torch.cat([audio_h, video_h, text_h], dim=-1)
        x = self.post_fusion_dropout(fusion_h)
        x = F.relu(self.post_fusion_layer_1(x), inplace=True)
        x = F.relu(self.post_fusion_layer_2(x), inplace=True)
        output = self.post_fusion_layer_3(x)

        res = {
            'Feature_T': text_h,
            'Feature_A': audio_h,
            'Feature_V': video_h,
            'Feature_M': fusion_h,
            'M': output
        }
        return res
        