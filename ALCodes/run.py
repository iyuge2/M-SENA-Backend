import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

# db interaction
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))

from app import db
from database import *
from constants import *

from load_data import MMDataLoader
from classifiers.AMIO import AMIO
from discriminators.ASIO import ASIO
from al_trains.ATIO import ATIO
from al_utils.functions import Storage

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(args):
    args.model_save_path = os.path.join(AL_CODES_PATH, 'save_models', \
                            f'{args.classifier}-{args.selector}-{args.datasetName}.pth')
    # load parameters
    with open(os.path.join(AL_CODES_PATH, 'config.json'), 'r') as f:
        config = json.load(f)
        classifier_config = config['classifiers'][args.classifier]['args']
        selector_config = config['selectors'][args.selector]['args']
        data_config = config['data'][args.datasetName]

    args = Storage(dict(vars(args), **classifier_config, **selector_config, **data_config))

    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    print("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    args.device = device
    # data
    dataloader = MMDataLoader(args)
    # do train
    model = AMIO(args).to(device)
    atio = ATIO().getTrain(args)
    atio.do_train(model, dataloader)
    # do test
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(device)
    outputs = atio.do_test(model, dataloader)
    # do selector
    asio = ASIO().getSelector(args)
    results = asio.do_select(outputs)

    # write
    label_by_dict = {
        "simple": 1,
        "middle": 2,
        "hard": 3
    }
    if args.use_db:
        # save results into database
        annotation_dict = {v:k for k, v in args.annotations.items()}
        if args.use_db:
            for k in ['simple', 'middle', 'hard']:
                ids, predicts = results[k]
                # print(ids)
                for i in range(len(ids)):
                    video_id, clip_id = ids[i].split('-')
                    sample = db.session.query(Dsample).filter_by(dataset_name=args.datasetName, \
                                            video_id=video_id, clip_id=clip_id).first()
                    sample.label_value = predicts[i]
                    sample.label_by = label_by_dict[k]
                    if k == 'simple':
                        sample.annotation = annotation_dict[predicts[i]]
        db.session.commit()
    else:
        # save results into label file
        df = pd.read_csv(args.label_path, encoding='utf-8', dtype={'video_id': str, 'clip_id':str})
        tmp_dict = {}
        for i in range(len(df)):
            video_id, clip_id = df.loc[i, ['video_id', 'clip_id']]
            tmp_dict[video_id + '-' + clip_id] = i
        for k in ['simple', 'middle', 'hard']:
            ids, predicts = results[k]
            for i in range(len(ids)):
                df.loc[tmp_dict[ids[i]], ['label', 'label_by']] = [predicts[i], label_by_dict[k]]
        df.to_csv(args.label_path, index=None, encoding='utf-8')
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_db', type=bool, default=False)
    parser.add_argument('--classifier', type=str, default='TFN')
    parser.add_argument('--selector', type=str, default='CONF')
    parser.add_argument('--datasetName', type=str, default='DemoDataset')
    parser.add_argument('--task_id', type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        if args.use_db:
            cur_task = db.session.query(Task).get(args.task_id)
        run(args)
        cur_task.state = 1
    except Exception as e:
        print(e)
        if args.use_db:
            cur_task = db.session.query(Task).get(args.task_id)
            cur_task.state = 2
    finally:
        if args.use_db:
            db.session.commit()