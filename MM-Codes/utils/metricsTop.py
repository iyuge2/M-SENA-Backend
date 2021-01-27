import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

__all__ = ['MetricsTop']

class MetricsTop():
    def __init__(self):
        self.metrics_dict = {
            'MOSI': self.__eval_mosi_regression,
            'RAW_MOSI': self.__eval_mosi_regression,
            'MOSEI': self.__eval_mosei_regression,
            'RAW_MOSEI': self.__eval_mosei_regression,
            'SIMS': self.__eval_sims_regression,
            'IEMOCAP': self.__eval_iemocap_classification
        }

    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth

        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def __eval_mosei_regression(self, y_pred, y_true):
        return self.__eval_sims_regression(y_pred, y_true)

    def __eval_mosi_regression(self, y_pred, y_true):
        return self.__eval_sims_regression(y_pred, y_true)

    def __eval_sims_regression(self, y_pred, y_true):
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()

        y_pred = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_pred, y_true)
        f_score = f1_score(y_pred, y_true, average='weighted')

        eval_results = {
            "Accuracy": round(accuracy, 4),
            "F1": round(f_score, 4),
        }
        return eval_results

    def __eval_iemocap_classification(self, y_true, y_pred):
        single = -1
        emos = ["Neutral", "Happy", "Sad", "Angry"]
        if single < 0:
            test_preds = y_pred.view(-1, 4, 2).cpu().detach().numpy()
            test_truth = y_true.view(-1, 4).cpu().detach().numpy()
            
            for emo_ind in range(4):
                print(f"{emos[emo_ind]}: ")
                test_preds_i = np.argmax(test_preds[:,emo_ind],axis=1)
                test_truth_i = test_truth[:,emo_ind]
                f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
                acc = accuracy_score(test_truth_i, test_preds_i)
                print("  - F1 Score: ", f1)
                print("  - Accuracy: ", acc)
        else:
            test_preds = y_pred.view(-1, 2).cpu().detach().numpy()
            test_truth = y_true.view(-1).cpu().detach().numpy()
            
            print(f"{emos[single]}: ")
            test_preds_i = np.argmax(test_preds,axis=1)
            test_truth_i = test_truth
            f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
            acc = accuracy_score(test_truth_i, test_preds_i)
            print("  - F1 Score: ", f1)
            print("  - Accuracy: ", acc)
        eval_results = {
            'F1_score': f1,
            'Mult_acc_2': acc 
        }
        return eval_results
    
    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]