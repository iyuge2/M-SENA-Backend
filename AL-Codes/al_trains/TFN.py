import os
import time
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from al_utils.functions import dict_to_str
from al_utils.metricsTop import MetricsTop

class TFN():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = MetricsTop().getMetics(args.datasetName)

    def do_train(self, model, dataloader):
        optimizer = optim.Adam(list(model.parameters())[2:], lr=self.args.learning_rate)
        # initilize results
        best_acc = 0
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred, y_true = [], []
            losses = []
            model.train()
            train_loss = 0.0
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device).view(-1).long()
                    # clear gradient
                    optimizer.zero_grad()
                    # forward
                    outputs = model(text, audio, vision)
                    # compute loss
                    loss = self.criterion(outputs['Predicts'], labels)
                    # backward
                    loss.backward()
                    # update
                    optimizer.step()
                    # store results
                    train_loss += loss.item()
                    y_pred.append(outputs['Predicts'].cpu())
                    y_true.append(labels.cpu())
            train_loss = train_loss / len(dataloader['train'])

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            train_results["Loss"] = train_loss
            print("TRAIN-(%s) (%d/%d)>> loss: %.4f " % (self.args.classifier, \
                epochs - best_epoch, epochs, train_loss) + dict_to_str(train_results))
            # validation
            val_results = self.do_valid(model, dataloader)
            
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
                return

    def do_valid(self, model, dataloader):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader['valid']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']["M"].to(self.args.device).view(-1).long()
                    outputs = model(text, audio, vision)
                    loss = self.criterion(outputs["Predicts"], labels)
                    eval_loss += loss.item()
                    y_pred.append(outputs["Predicts"].cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        results = self.metrics(pred, true)
        print("VALID-(%s)" % self.args.classifier + " >> loss: %.4f " % \
                eval_loss + dict_to_str(results))
        results["Loss"] = eval_loss

        return results
    
    def do_test(self, model, dataloader):
        model.eval()
        fields = ['ids', 'Feature_t', 'Feature_a', 'Feature_v', 'Feature_f', 'Predicts']
        results = {k: [] for k in fields}
        with torch.no_grad():
            with tqdm(dataloader['test']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    ids = batch_data['ids']
                    outputs = model(text, audio, vision)
                    for k in fields:
                        if k == 'ids':
                            results[k] += ids
                        else:
                            cur_res = outputs[k].detach().cpu()
                            results[k].append(cur_res)

        for k in fields:
            if k == 'ids':
                continue
            results[k] = torch.cat(results[k])

        return results