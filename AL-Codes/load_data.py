import os
import sys
import random
import pickle
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# db interaction
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))

from app import db
from database import *
from sqlalchemy import or_

__all__ = ['MMDataLoader']

class MMDataset(Dataset):
    def __init__(self, args, samples, mode='train'):
        self.mode = mode
        self.args = args
        self.samples = samples

        self.text_seq_len = MAX_TEXT_SEQ_LEN
        self.audio_seq_len = MAX_AUDIO_SEQ_LEN
        self.video_seq_len = MAX_VIDEO_SEQ_LEN

        DATA_MAP = {
            'DemoDataset': self.__init_demo,
        }
        DATA_MAP[args.datasetName](args)
        
    def __init_demo(self, args):
        with open(args.feature_path, 'rb') as f:
            data = pickle.load(f)
        self.text, self.vision, self.audio = [], [], []
        self.labels, self.ids = [], []
        for sample in self.samples:
            video_id, clip_id = sample
            key = video_id + '-' + clip_id
            self.text.append(data[key]['feature_T'])
            self.vision.append(data[key]['feature_V'])
            self.audio.append(data[key]['feature_A'])
            # self.labels.append(data[key]['label_M'])
            self.labels.append(int((data[key]['label_M'] + 1.0) * 1.49))
            self.ids.append(key)

        self.text = np.array(self.text)
        self.vision = np.array(self.vision)
        self.audio = np.array(self.audio)
        self.ids = self.ids

        self.__padding()

        self.labels = {
            'M': np.array(self.labels),
        }
        print(f"{self.mode} samples: {self.labels['M'].shape}")
        if 'normalized' in args.keys() and args.normalized:
            self.__normalize()

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # for visual and audio modality, we average across time
        # here the original data has shape (max_len, num_examples, feature_dim)
        # after averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
    
    def __padding(self):
        # NOTE: Here for dataset we manually cut the input into specific length.
        def Pad(modal_features, length):
            nsamples, seq_len, feature_dims = modal_features.shape
            if length == seq_len:
                return modal_features
            elif length < seq_len:
                return modal_features[:,:length,:]
            else:
                padding = np.zeros([nsamples, length-seq_len, feature_dims])
                modal_features = np.concatenate([modal_features, padding], axis=1)
                return modal_features
                       
        self.vision = Pad(self.vision, self.video_seq_len)
        self.text = Pad(self.text, self.text_seq_len)
        self.audio = Pad(self.audio, self.audio_seq_len)

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if self.args.use_bert:
            return [self.text.shape[2], self.audio.shape[1], self.vision.shape[1]]
        else:
            return [self.text.shape[1], self.audio.shape[1], self.vision.shape[1]]

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            # 'raw_text': self.rawText[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'ids': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 
        # if not self.args.aligned:
        #     sample['audio_lengths'] = self.audio_lengths[index]
        #     sample['vision_lengths'] = self.vision_lengths[index]
        return sample

def MMDataLoader(args):
    if args.use_db:
        samples = db.session.query(Dsample).filter_by(dataset_name=args.datasetName)
        train_samples = samples.filter(or_(Dsample.label_by==0, Dsample.label_by==1)).all()
        test_samples = samples.filter(or_(Dsample.label_by==-1, Dsample.label_by==2, Dsample.label_by==3)).all() 

        train_samples = [[sample.video_id, sample.clip_id] for sample in train_samples]
        test_samples = [[sample.video_id, sample.clip_id] for sample in test_samples]
    else:
        train_samples, test_samples = [], []
        df = pd.read_csv(args.label_path, encoding='utf-8', dtype={"video_id": "str", "clip_id": "str"})
        for i in range(len(df)):
            label_by, video_id, clip_id = df.loc[i, ['label_by', 'video_id', 'clip_id']]
            if label_by == 0 or label_by == 1:
                train_samples.append([video_id, clip_id])
            else:
                test_samples.append([video_id, clip_id])

    # split train / valid with 8:2
    train_samples, valid_samples = train_test_split(train_samples, test_size=args.valid_size, random_state=12345)


    datasets = {
        'train': MMDataset(args, train_samples, mode='train'),
        'valid': MMDataset(args, valid_samples, mode='valid'),
        'test': MMDataset(args, test_samples, mode='test'),
    }

    # if 'seq_lens' in args.keys():
    args.seq_lens = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    
    return dataLoader
