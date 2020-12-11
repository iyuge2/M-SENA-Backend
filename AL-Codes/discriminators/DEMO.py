import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class DEMO():
    def __init__(self, args):
        self.num_classes = args.num_classes
        self.high_score = 1 - 1 / (self.num_classes+2)
        self.low_score = self.high_score - 0.2

    def do_select(self, classifier_outputs):
        """
        classifier_outputs:
            Predicts: [nsamples, num_classes]
            Feature_t: [nsamples, text_dim]
            Feature_a: [nsamples, audio_dim]
            Feature_v: [nsamples, video_dim]
            Feature_f: [nsamples, fusion_dim]
        """
        probs = classifier_outputs['Predicts']
        probs = torch.softmax(probs, dim=1).numpy()
        ids = classifier_outputs['ids']

        max_probs = np.max(probs, axis=1)
        max_probs_ids = np.argmax(probs, axis=1)

        # simple samples ( max_prob > self.high_score)
        simple_samples_idx = np.where(max_probs > self.high_score)[0].tolist()
        simple_ids = [ids[v] for v in simple_samples_idx]
        simple_results = [simple_ids, max_probs_ids[simple_samples_idx].tolist()]

        # middle samples ( max_prob <= self.high_score and max_prob > self.low_score)
        middle_samples_idx = np.where((max_probs <= self.high_score) & (max_probs > self.low_score))[0].tolist()
        middle_ids = [ids[v] for v in middle_samples_idx]
        middle_results = [middle_ids, max_probs_ids[middle_samples_idx].tolist()]

        # hard samples ( max_prob <= self.low_score)
        hard_samples_idx = np.where(max_probs <= self.low_score)[0].tolist()
        hard_ids = [ids[v] for v in hard_samples_idx]
        hard_results = [hard_ids, max_probs_ids[hard_samples_idx].tolist()]

        # design a algorithm to select the typical hard samples from hard samples
        # move typical hard samples -> hard samples
        # move atypical hard samples -> middle samples
        # in this algotithm, supposing that all hard samples are typical hard samples

        # output
        res = {
            'simple': simple_results,
            'middle': middle_results,
            'hard': hard_results
        }

        return res