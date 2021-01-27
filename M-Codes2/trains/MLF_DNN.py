import os
import time
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop

class MLF_DNN():
    def __init__(self, args):
        assert args.datasetName == 'SIMS'

        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = MetricsTop().getMetics(args.datasetName)

    def do_train(self, model, dataloader):
        optimizer = optim.Adam([{"params": list(model.Model.text_subnet.parameters()), "weight_decay": self.args.text_weight_decay},
                                {"params": list(model.Model.audio_subnet.parameters()), "weight_decay": self.args.audio_weight_decay},
                                {"params": list(model.Model.video_subnet.parameters()), "weight_decay": self.args.video_weight_decay}],
                                lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        # initilize results
        best_acc = 0
        epochs, best_epoch = 0, 0
        epoch_results = {
            'train': [],
            'valid': [],
            'test': []
        }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}
            losses = []
            model.train()
            train_loss = 0.0
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']
                    for k in labels.keys():
                        labels[k] = labels[k].to(self.args.device).view(-1).long()
                    # clear gradient
                    optimizer.zero_grad()
                    # forward
                    outputs = model(text, audio, vision)
                    # compute loss
                    loss = 0.0
                    for m in self.args.tasks:
                        loss += eval('self.args.'+m) * self.criterion(outputs[m], labels[m])
                    # backward
                    loss.backward()
                    # update
                    optimizer.step()
                    # store results
                    train_loss += loss.item()
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].detach().cpu())
                        y_true[m].append(labels['M'].detach().cpu())
            train_loss = train_loss / len(dataloader['train'])
            print("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                        epochs - best_epoch, epochs, self.args.cur_time, train_loss))
            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results = self.metrics(pred, true)
                print('%s: >> ' %(m) + dict_to_str(train_results))
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            test_results = self.do_test(model, dataloader['test'], mode="TEST")
            epoch_results['valid'].append(val_results)
            epoch_results['test'].append(test_results)

            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= best_valid if min_or_max == 'min' else cur_valid >= best_valid
            # save best model
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results

    def do_test(self, model, dataloader, mode="VAL", need_details=False):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        eval_loss = 0.0
        if need_details:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_T": [],
                "Feature_A": [],
                "Feature_V": [],
                "Feature_M": [],
            }
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']
                    for k in labels.keys():
                        labels[k] = labels[k].to(self.args.device).view(-1).long()
                    outputs = model(text, audio, vision)

                    if need_details:
                        ids.extend(batch_data['id'])
                        for item in features.keys():
                            features[item].append(outputs[item].cpu().detach().numpy())
                        all_labels.extend(labels['M'].cpu().detach().tolist())
                        preds = outputs["M"].cpu().detach().numpy()
                        test_preds_i = np.argmax(preds, axis=1)
                        sample_results.extend(test_preds_i)

                    loss = 0.0
                    for m in self.args.tasks:
                        loss += eval('self.args.'+m) * self.criterion(outputs[m], labels[m])
                    eval_loss += loss.item()
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].detach().cpu())
                        y_true[m].append(labels['M'].detach().cpu())
        eval_loss = eval_loss / len(dataloader)
        print(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        for i, m in enumerate(self.args.tasks):
            pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
            results = self.metrics(pred, true)
            print('%s: >> ' %(m) + dict_to_str(results))
            if i == 0:
                eval_results = results
                eval_results["Loss"] = round(eval_loss, 4)
        if need_details:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels
        return eval_results