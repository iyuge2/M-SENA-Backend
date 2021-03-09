import os
import sys
import time
import json
import random
# import logging
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from datetime import datetime
from sklearn.decomposition import PCA

# db interaction
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

# from config.config_run import Config
# from config.config_debug import ConfigDebug
from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader
from data.livePre import MLive
from utils.functions import Storage

from app import db
from database import *
from constants import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(args):
    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    print("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    args.device = device
    # data
    dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)

    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
                # print(p)
        return answer
    print(f'The model has {count_parameters(model)} trainable parameters')
    # exit()
    # using multiple gpus
    if using_cuda and len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model,
                                      device_ids=args.gpu_ids,
                                      output_device=args.gpu_ids[0])
    # start running
    # do train
    atio = ATIO().getTrain(args)
    # do train
    epoch_results = atio.do_train(model, dataloader)
    # load pretrained model
    assert os.path.exists(args.model_save_path)
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(device)
    # do test
    final_results = {}
    final_results['train'] = atio.do_test(model, dataloader['train'], mode="TRAIN", need_details=True)
    final_results['valid'] = atio.do_test(model, dataloader['valid'], mode="VALID", need_details=True)
    final_results['test'] = atio.do_test(model, dataloader['test'], mode="TEST", need_details=True)
    
    # don't save pretrained model for debug mode
    if args.run_mode == 'Tune':
        os.remove(args.model_save_path)

    return {"epoch_results": epoch_results, 'final_results': final_results}

def run_train(model_args, args):
    if not os.path.exists(MODEL_TMP_SAVE):
        os.makedirs(MODEL_TMP_SAVE)

    result = Result(
            dataset_name=args.datasetName,
            model_name=args.modelName,
            is_tuning=args.run_mode,
            args=json.dumps(model_args),
            save_model_path='',
            loss_value=0.0,
            accuracy=0.0,
            f1=0.0,
            description=args.description
        )
    db.session.add(result)
    db.session.flush()

    args.model_save_path = os.path.join(MODEL_TMP_SAVE,\
                                        f'{args.modelName}-{args.datasetName}-{result.result_id}.pth')

    try:
        # run results
        print('Start running %s...' %(args.modelName))
        print(args)
        # runnning
        setup_seed(args.seed)
        model_results = run(args)
        
        # sample id: video_id$_$clip_id
        sample_dict = {}
        samples = db.session.query(Dsample).filter_by(dataset_name=args.datasetName).all()
        for sample in samples:
            key = sample.video_id + '$_$' + sample.clip_id
            sample_dict[key] = [sample.sample_id, sample.annotation]
        annotation_dict = {v:k for k, v in args.annotations.items()}

        # update final results of test set
        result.loss_value = model_results['final_results']['test']['Loss']
        result.accuracy = model_results['final_results']['test']['Accuracy']
        result.f1 = model_results['final_results']['test']['F1']

        # save features
        if args.run_mode == 'Train':
            print("feature pca ...")
            features = {
                k: model_results['final_results'][k]['Features']
                for k in ['train', 'valid', 'test']
            }
            all_features = {}
            for select_modes in [['train', 'valid', 'test'], ['train', 'valid'], ['train', 'test'], \
                                ['valid', 'test'], ['train'], ['valid'], ['test']]:
                # create label index dict
                # {"Negative": [1,2,5,...], "Positive": [...], ...}
                cur_labels = []
                for mode in select_modes:
                    cur_labels.extend(model_results['final_results'][mode]['Labels'])
                cur_labels = np.array(cur_labels)
                label_index_dict = {}
                for k, v in args.annotations.items():
                    label_index_dict[k] = np.where(cur_labels == v)[0].tolist()
                # handle features
                cur_mode_features_2d = {}
                cur_mode_features_3d = {}
                for name in ['Feature_T', 'Feature_A', 'Feature_V', 'Feature_M']: 
                    cur_features = []
                    for mode in select_modes:
                        cur_features.append(features[mode][name])
                    cur_features = np.concatenate(cur_features, axis=0)
                    # PCA analysis
                    pca=PCA(n_components=3, whiten=True)
                    features_3d = pca.fit_transform(cur_features)
                    # split by labels
                    cur_mode_features_3d[name] = {}
                    for k, v in label_index_dict.items():
                        cur_mode_features_3d[name][k] = features_3d[v].tolist()
                    # PCA analysis
                    pca=PCA(n_components=2, whiten=True)
                    features_2d = pca.fit_transform(cur_features)
                    # split by labels
                    cur_mode_features_2d[name] = {}
                    for k, v in label_index_dict.items():
                        cur_mode_features_2d[name][k] = features_2d[v].tolist()
                all_features['-'.join(select_modes)] = {'2D': cur_mode_features_2d, '3D': cur_mode_features_3d}
            # save features
            save_path = os.path.splitext(args.model_save_path)[0] + '.pkl'
            with open(save_path, 'wb') as fp:
                pickle.dump(all_features, fp, protocol = 4)
            result.save_model_path = args.model_save_path
            print(f'feature saving at {save_path}...')

        # update sample results
        for mode in ['train', 'valid', 'test']:
            final_results = model_results['final_results'][mode]
            for i, cur_id in enumerate(final_results["Ids"]):
                payload = SResults(
                    result_id=result.result_id,
                    sample_id=sample_dict[cur_id][0],
                    label_value=sample_dict[cur_id][1],
                    predict_value=annotation_dict[final_results["SResults"][i]]
                )
                db.session.add(payload)
        
        # update epoch results
        # only save epoch results in the first seed 
        cur_results = {}
        for mode in ['train', 'valid', 'test']:
            cur_epoch_results = model_results['final_results'][mode]
            cur_results[mode] = {
                "loss_value":cur_epoch_results["Loss"],
                "accuracy":cur_epoch_results["Accuracy"],
                "f1":cur_epoch_results["F1"]
            }
        payload = EResult(
            result_id=result.result_id,
            epoch_num=-1,
            results=json.dumps(cur_results)
        )
        db.session.add(payload)

        epoch_num = len(model_results['epoch_results']['train'])
        for i in range(1, epoch_num+1):
            cur_results = {}
            for mode in ['train', 'valid', 'test']:
                cur_epoch_results = model_results['epoch_results'][mode][i-1]
                cur_results[mode] = {
                    "loss_value":cur_epoch_results["Loss"],
                    "accuracy":cur_epoch_results["Accuracy"],
                    "f1":cur_epoch_results["F1"]
                }
            payload = EResult(
                result_id=result.result_id,
                epoch_num=i,
                results=json.dumps(cur_results)
            )
            db.session.add(payload)
        db.session.commit()

    except Exception as e:
        print(e)
        db.session.rollback() # 回滚操作
        # remove saved features
        save_paths = args.model_save_path.split('.')[0]
        save_paths = glob(save_paths + '.*')
        for save_path in save_paths:
            if os.path.exists(save_path):
                os.remove(save_path)
        raise Exception(e)

def run_pre(args):
    def load_model_params():
        # load model config
        if args.parameters == "":
            with open(os.path.join(MM_CODES_PATH, 'config.json'), 'r') as f:
                model_config = json.load(f)["MODELS"]
                model_config = model_config[args.modelName]['args'][args.datasetName]
        else:
            model_config = json.loads(args.parameters)
        return model_config

    def load_data_params():
        # load data config
        with open(os.path.join(MM_CODES_PATH, 'config.json'), 'r') as f:
            data_config = json.load(f)["DATASETS"][args.datasetName]
        if "need_data_aligned" in model_args and model_args['need_data_aligned']:
            data_config = data_config['aligned']
        else:
            data_config = data_config['unaligned']
        data_config['feature_path'] = os.path.join(DATASET_ROOT_DIR, data_config['feature_path'])
        return data_config

    if args.run_mode == 'Tune':
        for _ in range(args.tune_times):
            model_args = load_model_params()
            # select a random param
            for k, v in model_args.items():
                if k == "gpu_ids":
                    continue
                if isinstance(v, list):
                    model_args[k] = random.choice(v)
            # skip the repeated parameters
            existed = db.session.query(Result).filter_by(dataset_name=args.datasetName, \
                            model_name=args.modelName, args=json.dumps(model_args)).first()
            if not existed:
                data_args = load_data_params()
                new_args = Storage(dict(vars(args), **model_args, **data_args))
                run_train(model_args, new_args)
    elif args.run_mode == 'Train':
        model_args = load_model_params()
        data_args = load_data_params()
        new_args = Storage(dict(vars(args), **model_args, **data_args))
        run_train(model_args, new_args)
    else:
       pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', type=str, default="Train",
                        help='Tune; Train')
    parser.add_argument('--modelName', type=str, default="MTFN",
                        help='support TFN/LMF/MFN/EF_LSTM/LF_DNN/Graph-MFN/MTFN/MLF_DNN/MLMF/MULT/MISA')
    parser.add_argument('--datasetName', type=str, default='SIMS',
                        help='support SIMS/MOSI/MOSEI')
    parser.add_argument('--parameters', type=str, default='')
    parser.add_argument('--task_id', type=int)
    parser.add_argument('--tune_times', type=int, default=20)
    parser.add_argument('--description', type=str, default='')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    try:
        cur_task = db.session.query(Task).get(args.task_id)
        run_pre(args)
        cur_task.state = 1
    except Exception as e:
        print(e)
        cur_task.state = 2
    finally:
        cur_task.end_time = datetime.utcnow() + timedelta(hours=8)
        db.session.commit()