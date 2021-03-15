import os
import sys
import time
import json
import torch
import random
import numpy as np

# db interaction
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

# from config.config_run import Config
# from config.config_debug import ConfigDebug
from models.AMIO import AMIO
from data.livePre import MLive
from utils.functions import Storage

from constants import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x.tolist()

def run_live(args):
    print(args)
    args['text_language'] = args['language']
    args.pop('language', None)
    # load args
    with open(os.path.join(MM_CODES_PATH, 'config.json'), 'r') as fp:
        config = json.load(fp)
    model_args = config["MODELS"][args['modelName']]['args'][args['datasetName']]
    dataset_args = config["DATASETS"][args['datasetName']]
    if "need_data_aligned" in model_args and model_args['need_data_aligned']:
        dataset_args = dataset_args['aligned']
    else:
        dataset_args = dataset_args['unaligned']
    # load args
    args = Storage(dict(**model_args, **dataset_args, **args))
    args.device = 'cpu'
    print(args)
    # load model
    setup_seed(args.seed)
    model = AMIO(args)
    pretrained_model_path = os.path.join(MODEL_TMP_SAVE, args.pre_trained_model)
    # if os.path.exists(pretrained_model_path):
    model.load_state_dict(torch.load(pretrained_model_path))
    
    # data pre
    dp = MLive(args.live_working_dir, args.transcript, args.text_language)
    dp.dataPre()
    text, audio, video = dp.getEmbeddings(args.seq_lens, args.feature_dims)
    text = torch.Tensor(text).unsqueeze(0)
    audio = torch.Tensor(audio).unsqueeze(0)
    video = torch.Tensor(video).unsqueeze(0)
    print(text.size(), audio.size(), video.size())

    if args.need_normalized:
        audio = torch.mean(audio, dim=1, keepdims=True)
        video = torch.mean(video, dim=1, keepdims=True)
    # predict
    model.eval()
    with torch.no_grad():
        outputs = model(text, audio, video)

    if 'tasks' not in args:
        args.tasks = 'M'

    annotation_dict = {v:k for k, v in args.annotations.items()}

    ret = {}
    for m in args.tasks:
        cur_output = outputs[m].detach().squeeze().numpy()
        cur_output = np.clip(cur_output, 1e-8, 10)
        cur_output = softmax(cur_output)
        sentiment = np.argmax(cur_output)
        # json cannot serialize float32
        probs = {annotation_dict[i]: str(round(v, 4)) for i, v in enumerate(cur_output)}

        ret[m] = {
            'model': args.modelName,
            "probs": probs
        }

    print(ret)
    return ret

if __name__ == "__main__":
    pre_trained_model = 'MLF_DNN-SIMS-107.pth'
    model_name, dataset_name = pre_trained_model.split('-')[0:2]
    other_args = {
        'pre_trained_model': pre_trained_model,
        'modelName': model_name,
        'datasetName': dataset_name,
        'live_working_dir': os.path.join(MM_CODES_PATH, 'tmp_dir/1614251422871'),
        'transcript': "这个苹果有问题，不好吃",
        'language': "Chinese"
    }
    run_live(other_args)