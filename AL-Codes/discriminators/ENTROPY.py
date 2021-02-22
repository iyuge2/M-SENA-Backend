import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ENTROPY():
    def __init__(self, args):
        self.sift = args.sift
        self.num_classes = args.num_classes
        self.high_score = 0.9
        self.low_score = 0.7
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

        max_probs_ids = np.argmax(probs, axis=1)


        info = []
        for line in probs:
            add = 0
            for i, value in enumerate(line):
                add += value*math.log(value)
            info.append(-add)
        entropy = np.array(info)

        if self.sift == 'threshold':
            simple_samples_idx = np.where(entropy < self.low_score)[0].tolist()
            simple_ids = [ids[v] for v in simple_samples_idx]
            simple_results = [simple_ids, max_probs_ids[simple_samples_idx].tolist()]

            # middle samples ( max_prob <= self.high_score and max_prob > self.low_score)
            middle_samples_idx = np.where((entropy <= self.high_score) & (entropy > self.low_score))[0].tolist()
            middle_ids = [ids[v] for v in middle_samples_idx]
            middle_results = [middle_ids, max_probs_ids[middle_samples_idx].tolist()]

            # hard samples ( max_prob <= self.low_score)
            hard_samples_idx = np.where(entropy > self.high_score)[0].tolist()
            hard_ids = [ids[v] for v in hard_samples_idx]
            hard_results = [hard_ids, max_probs_ids[hard_samples_idx].tolist()]
        
        elif self.sift == 'ratio':
            length = len(entropy)
            entropy_sort = np.sort(entropy).tolist()
            index_probs = np.argsort(entropy)
            sort_porbs_ids = np.array([max_probs_ids[i] for i in index_probs]).tolist()
            sort_ids = np.array([ids[i] for i in index_probs]).tolist()

            middle = int(length*self.middle_rate)
            hard = int(length*self.hard_rate)

            simple_ids = sort_ids[0:middle]
            simple_results = [simple_ids, sort_porbs_ids[0:middle]]

            middle_ids = sort_ids[middle:hard]
            middle_results = [middle_ids, sort_porbs_ids[middle:hard]]

            hard_ids = sort_ids[hard:length]
            hard_results = [hard_ids, sort_porbs_ids[hard:length]]

        # output
        res = {
            'simple': simple_results,
            'middle': middle_results,
            'hard': hard_results
        }

        return res