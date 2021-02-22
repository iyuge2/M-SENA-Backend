import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MARGIN():
    def __init__(self, args):
        self.sift = args.sift
        self.num_classes = args.num_classes
        self.high_score = 0.6
        self.low_score = 0.2
        self.hard_rate = 0.85
        self.middle_rate = 0.2

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
        for i in range(len(probs)):
            probs[i][max_probs_ids[i]] = 0
        second_probs = np.max(probs, axis=1)
        margin = np.array([max_probs[i] - second_probs[i] for i in range(len(max_probs))])

        if self.sift == 'threshold':
            # simple samples ( max_prob > self.high_score)
            simple_samples_idx = np.where(margin > self.high_score)[0].tolist()
            simple_ids = [ids[v] for v in simple_samples_idx]
            simple_results = [simple_ids, max_probs_ids[simple_samples_idx].tolist()]

            # middle samples ( max_prob <= self.high_score and max_prob > self.low_score)
            middle_samples_idx = np.where((margin <= self.high_score) & (max_probs > self.low_score))[0].tolist()
            middle_ids = [ids[v] for v in middle_samples_idx]
            middle_results = [middle_ids, max_probs_ids[middle_samples_idx].tolist()]

            # hard samples ( max_prob <= self.low_score)
            hard_samples_idx = np.where(margin <= self.low_score)[0].tolist()
            hard_ids = [ids[v] for v in hard_samples_idx]
            hard_results = [hard_ids, max_probs_ids[hard_samples_idx].tolist()]
        
        elif self.sift == 'ratio':
            length = len(max_probs)
            margin_sort = np.sort(margin).tolist()
            index_probs = np.argsort(margin)
            sort_porbs_ids = np.array([max_probs_ids[i] for i in index_probs]).tolist()
            sort_ids = np.array([ids[i] for i in index_probs]).tolist()

            middle = int(length*(1 - self.middle_rate))
            hard = int(length*(1 - self.hard_rate))

            hard_ids = sort_ids[0:hard]
            hard_results = [hard_ids, sort_porbs_ids[0:hard]]

            # middle samples ( max_prob <= self.high_score and max_prob > self.low_score)
            middle_ids = sort_ids[hard:middle]
            middle_results = [middle_ids, sort_porbs_ids[hard:middle]]

            # hard samples ( max_prob <= self.low_score)
            simple_ids = sort_ids[middle:length]
            simple_results = [simple_ids, sort_porbs_ids[middle:length]]


        # output
        res = {
            'simple': simple_results,
            'middle': middle_results,
            'hard': hard_results
        }

        return res