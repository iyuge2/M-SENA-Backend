import os
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

__all__ = ['Config']

class Storage(dict):
    """
    A Storage object is like a dictionary except `obj.foo` can be used inadition to `obj['foo']`
    ref: https://blog.csdn.net/a200822146085/article/details/88430450
    """
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __str__(self):
        return "<" + self.__class__.__name__ + dict.__repr__(self) + ">"

class Config():
    def __init__(self, args):
        # parameters for data
        # global parameters for running
        self.globalArgs = args
        # hyper parameters for models
        self.HYPER_MODEL_MAP = {
            # single-task
            'tfn': self.__TFN,
            'lmf': self.__LMF,
            'mfn': self.__MFN,
            'ef_lstm': self.__EF_LSTM,
            'lf_dnn': self.__LF_DNN,
            'graph_mfn': self.__Graph_MFN,
            'bert_mag': self.__BERT_MAG,
            'bert_mult': self.__BERT_MULT,
            'bert_misa': self.__BERT_MISA,
            # multi-task
            'mtfn': self.__MTFN,
            'mlmf': self.__MLMF,
            'mlf_dnn': self.__MLF_DNN,
            # newModel
            'dmmsa': self.__DMMSA,
            'lsmmsa': self.__LSMMSA
        }
        # hyper parameters for datasets
        self.HYPER_DATASET_MAP = self.__datasetCommonParams()
    
    def __datasetCommonParams(self):
        assert self.globalArgs.datasetName in ['mosi', 'mosei', 'sims']

        tmp = "aligned" if self.globalArgs.aligned else "unaligned"
        if self.globalArgs.datasetName in ['mosi', 'mosei']:
            text_len = 50
        elif self.globalArgs.datasetName in ['sims']:
            text_len = 39
            
        dataPath = os.path.join(self.globalArgs.data_dir, self.globalArgs.datasetName, \
                                tmp + '_' + str(text_len) + '.pkl')
        tmp = {
            'mosi':{
                'dataPath': dataPath,
                'seq_lens': (50, 50, 50) if self.globalArgs.aligned else (50, 500, 375),
                # (text, audio, video)
                'feature_dims': (768, 5, 20),
                'train_samples': 1284,
                'language': 'en',
                'KeyEval': 'Loss'
            },
            'mosei':{
                'dataPath': dataPath,
                'seq_lens': (50, 50, 50) if self.globalArgs.aligned else (50, 500, 375),
                # (text, audio, video)
                'feature_dims': (768, 74, 35),
                'train_samples': 16326,
                'language': 'en',
                'KeyEval': 'Loss'
            },
            'sims':{
                'dataPath': dataPath,
                # (batch_size, seq_lens, feature_dim)
                'seq_lens': (39, 400, 55), # (text, audio, video)
                'feature_dims': (768, 33, 709), # (text, audio, video)
                'train_samples': 1368,
                'language': 'cn',
                'KeyEval': 'Loss',
            },
        }
        return tmp

    def __BERT_MULT(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'early_stop': 6,
                'use_finetune': False,
                'use_bert': False,
                # Task
                'num_classes': 1, # compute regression
                'aligned': True, # consider aligned experiment or not
                # Architecture
                'attn_mask': True, # use attention mask for Transformer
                'attn_dropout_a': 0.2,
                'attn_dropout_v': 0.2,
                'relu_dropout': 0.1,
                'embed_dropout': 0.2,
                'res_dropout': 0.1,
                # Tuning
                'patience': 20, # when to decay learning rate (default: 20)
                # Logistics
                'weight_decay': 0.0,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'criterion': 'L1Loss',
                    #  transformers hidden unit size(d) &&  transformers hidden unit size(d)
                    'dst_feature_dim_nheads': (40, 10), 
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'update_epochs': 8,
                    'batch_size': 8,
                    'learning_rate': 1e-3,
                    'nlevels': 4, # number of layers(Blocks) in the Crossmodal Networks
                    # temporal convolution kernel size
                    'conv1d_kernel_size_l': 1, 
                    'conv1d_kernel_size_a': 3,
                    'conv1d_kernel_size_v': 3,
                    # dropout
                    'text_dropout': 0.2, # textual Embedding Dropout
                    'attn_dropout': 0.2, # crossmodal attention block dropout
                    'output_dropout': 0.1,
                    'grad_clip': 0.8, # gradient clip value (default: 0.8)
                },
                'mosei':{
                    'criterion': 'L1Loss',
                    #  transformers hidden unit size(d) &&  transformers hidden unit size(d)
                    'dst_feature_dim_nheads': (40, 10), 
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'update_epochs': 1,
                    'batch_size': 8,
                    'learning_rate': 1e-3,
                    'nlevels': 4, # number of layers(Blocks) in the Crossmodal Networks
                    # temporal convolution kernel size
                    'conv1d_kernel_size_l': 1, 
                    'conv1d_kernel_size_a': 3,
                    'conv1d_kernel_size_v': 3,
                    # dropout
                    'text_dropout': 0.2, # textual Embedding Dropout
                    'attn_dropout': 0.2, # crossmodal attention block dropout
                    'output_dropout': 0.1,
                    'grad_clip': 0.8, # gradient clip value (default: 0.8)
                },
                'sims':{
                    'criterion': 'L1Loss',
                    #  transformers hidden unit size(d) &&  transformers hidden unit size(d)
                    'dst_feature_dim_nheads': (40, 10),
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'update_epochs': 1,
                    'batch_size': 32,
                    'learning_rate': 1e-3,
                    'nlevels': 4, # number of layers(Blocks) in the Crossmodal Networks
                    # temporal convolution kernel size
                    'conv1d_kernel_size_l': 1, 
                    'conv1d_kernel_size_a': 3,
                    'conv1d_kernel_size_v': 3,
                    # dropout
                    'text_dropout': 0.2, # textual Embedding Dropout
                    'attn_dropout': 0.2, # crossmodal attention block dropout
                    'output_dropout': 0.1,
                    'grad_clip': 0.8, # gradient clip value (default: 0.8)
                }
            },
        }
        return tmp
    
    def __BERT_MAG(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'use_finetune': True,
                'use_bert': True,
                'early_stop': 12,
                'multi_label': False,
                'need_normalize': False,
                # Tuning
                'weight_decay': 0.0,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # 'num_layers': 2,
                    # 'dropout': 0.1,
                    # ref Original Paper
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'update_epochs': 1,
                    'batch_size': 32,
                    'learning_rate': 5e-5,
                    'loss_function':'ll1',
                    'd_acoustic_in':5,
                    'd_visual_in':20,
                    # 'h_audio_lstm':16,
                    # 'h_video_lstm':16,
                    'h_merge_sent':768,
                    # 'fc1_out':32,
                    # 'fc1_dropout':0.1,
                    'hidden_dropout_prob':0.2,
                    'beta_shift':1,
                    # add MAG after "AV_index" layer
                    # -1 means adding after all layers
                    # -2 means not use MAG
                    'AV_index':1,
                    'output_mode':'regression',
                    'num_labels':1, # is valid when output_mode == "classification"
                },
                'mosei':{
                    # 'num_layers': 2,
                    # 'dropout': 0.1,
                    # ref Original Paper
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'update_epochs': 1,
                    'batch_size': 32,
                    'learning_rate': 5e-5,
                    'loss_function':'ll1',
                    'd_acoustic_in':74,
                    'd_visual_in':35,
                    # 'h_audio_lstm':16,
                    # 'h_video_lstm':16,
                    'h_merge_sent':768,
                    # 'fc1_out':32,
                    # 'fc1_dropout':0.1,
                    'hidden_dropout_prob':0.2,
                    'beta_shift':1,
                    # add MAG after "AV_index" layer
                    # -1 means adding after all layers
                    # -2 means not use MAG
                    'AV_index':1,
                    'output_mode':'regression',
                    'num_labels':1, # is valid when output_mode == "classification"
                },
                'sims':{
                    # 'num_layers': 2,
                    # 'dropout': 0.1,
                    # ref Original Paper
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'update_epochs': 1,
                    'batch_size': 32,
                    'learning_rate': 5e-5,
                    'loss_function':'ll1',
                    'd_acoustic_in':33,
                    'd_visual_in':709,
                    # 'h_audio_lstm':16,
                    # 'h_video_lstm':16,
                    'h_merge_sent':768,
                    # 'fc1_out':32,
                    # 'fc1_dropout':0.1,
                    'hidden_dropout_prob':0.2,
                    'beta_shift':1,
                    # add MAG after "AV_index" layer
                    # -1 means adding after all layers
                    # -2 means not use MAG
                    'AV_index':7,
                    'output_mode':'regression',
                    'num_labels':2, # is valid when output_mode == "classification"
                },
            },
        }
        return tmp
    
    def __BERT_MISA(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'use_finetune': True,
                'use_bert': True,
                'early_stop': 12,
                'multi_label': True,
                'need_normalize': False,
                'weight_decay': 0.0,
                'rnncell': 'lstm',
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'update_epochs': 1,
                    'batch_size': 64,
                    'use_cmd_sim': True,
                    'num_classes': 1,
                    'learning_rate': 1e-4,
                    'grad_clip': 1.0,
                    'hidden_size': 128,
                    'dropout': 0.5,
                    'reverse_grad_weight': 1.0,
                    'diff_weight': 0.3,
                    'sim_weight': 1.0,
                    'sp_weight': 0.0,
                    'recon_weight': 1.0,
                },
                'mosei':{
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'update_epochs': 1,
                    'batch_size': 16,
                    'use_cmd_sim': True,
                    'num_classes': 1,
                    'learning_rate': 1e-4,
                    'grad_clip': 1.0,
                    'hidden_size': 128,
                    'dropout': 0.1,
                    'reverse_grad_weight': 1.0,
                    'diff_weight': 0.3,
                    'sim_weight': 0.7,
                    'sp_weight': 0.0,
                    'recon_weight': 0.7,
                },
                'sims':{
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'update_epochs': 1,
                    'batch_size': 8,
                    'use_cmd_sim': True,
                    'num_classes': 1,
                    'learning_rate': 1e-4,
                    'grad_clip': 1.0,
                    'hidden_size': 128,
                    'dropout': 0.5,
                    'reverse_grad_weight': 1.0,
                    'diff_weight': 0.3,
                    'sim_weight': 1.0,
                    'sp_weight': 0.0,
                    'recon_weight': 1.0,
                },
            },
        }
        return tmp
    

    def __TFN(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'need_normalize': True,
                # Tuning
                'early_stop': 20,
                # Logistics
                'weight_decay': 0.0,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'hidden_dims': (256, 32, 256),
                    'text_out': 64,
                    'post_fusion_dim': 16,
                    'dropouts': (0.2, 0.2, 0.2, 0.3),
                    'criterion': 'L1Loss',
                    'num_classes': 1, # compute regression
                    # ref Original Paper
                    'batch_size': 32,
                    'learning_rate': 5e-4,
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
                'sims':{
                    'hidden_dims': (128,16,128),
                    'text_out': 128,
                    'post_fusion_dim': 64,
                    'dropouts': (0.2,0.2,0.2,0.2),
                    'criterion': 'L1Loss',
                    'num_classes': 1, # compute regression
                    # ref Original Paper
                    'batch_size': 128,
                    'learning_rate': 2e-3,
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                }
            },
        }
        return tmp
    
    def __LMF(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'need_normalize': True,
                # Tuning
                'early_stop': 20,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'hidden_dims': (64, 16, 64),
                    'dropouts': (0.3, 0.3, 0.3, 0.5),
                    'output_dim': 1,
                    'criterion': 'L1Loss',
                    'rank': 5,
                    'use_softmax': False,
                    # ref Original Paper
                    'batch_size': 64,
                    'learning_rate': 2e-3,
                    'factor_lr': 1e-4, # factor_learning_rate
                    'weight_decay': 1e-4,
                    'grad_clip': 0.0, # gradient clip value
                },
                'sims':{
                    'hidden_dims': (128, 16, 128),
                    'dropouts': (0.3, 0.3, 0.3, 0.5),
                    'output_dim': 1,
                    'criterion': 'L1Loss',
                    'rank': 4,
                    'use_softmax': False,
                    # ref Original Paper
                    'batch_size': 128,
                    'learning_rate': 1e-3,
                    'factor_lr': 5e-4, # factor_learning_rate
                    'weight_decay': 1e-3,
                    'grad_clip': 0.0, # gradient clip value
                },
            },
        }
        return tmp
    
    def __MFN(self):
        tmp = {
            'commonParas':{
                'need_align': True,
                # Tuning
                'early_stop': 20,
                # Logistics
                'weight_decay': 0.0,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'hidden_dims': (128, 4, 16),
                    'memsize': 64,
                    'windowsize': 2,
                    'output_dim': 1,
                    'NN1Config': {"drop": 0.7, "shapes": 256},
                    'NN2Config': {"drop": 0.7, "shapes": 128},
                    'gamma1Config': {"drop": 0.7, "shapes": 128},
                    'gamma2Config': {"drop": 0.2, "shapes": 32},
                    'outConfig': {"drop": 0.5, "shapes": 128},
                    'criterion': 'L1Loss',
                    # ref Original Paper
                    'batch_size': 16,
                    'learning_rate': 1e-3,
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
                'sims':{
                    'hidden_dims': (128, 16, 128),
                    'memsize': 64,
                    'windowsize': 2,
                    'output_dim': 1,
                    'NN1Config': {"drop": 0.2, "shapes": 32},
                    'NN2Config': {"drop": 0.7, "shapes": 128},
                    'gamma1Config': {"drop": 0.7, "shapes": 256},
                    'gamma2Config': {"drop": 0.7, "shapes": 32},
                    'outConfig': {"drop": 0.2, "shapes": 32},
                    'criterion': 'L1Loss',
                    # ref Original Paper
                    'batch_size': 128,
                    'learning_rate': 5e-4,
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
            },
        }
        return tmp
    
    def __EF_LSTM(self):
        tmp = {
            'commonParas':{
                'need_align': True,
                # Tuning
                'early_stop': 20,
                # Logistics
                'weight_decay': 0.0,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'hidden_dims': 64,
                    'output_dim': 1,
                    'criterion': 'L1Loss',
                    'num_layers': 2,
                    'dropout': 0.3,
                    # ref Original Paper
                    'batch_size': 16,
                    'learning_rate': 1e-3,
                },
                'sims':{
                    'hidden_dims': 16,
                    'output_dim': 1,
                    'criterion': 'L1Loss',
                    'num_layers': 2,
                    'dropout': 0.4,
                    # ref Original Paper
                    'batch_size': 128,
                    'learning_rate': 1e-3,
                },
            },
        }
        return tmp
    
    def __LF_DNN(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'need_normalize': True,
                # Tuning
                'early_stop': 20,
                'patience': 0, # when to decay learning rate
                # Logistics
                'weight_decay': 0.0,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'hidden_dims': (128, 16, 128),
                    'text_out': 32,
                    'post_fusion_dim': 32,
                    'dropouts': (0.2, 0.2, 0.2, 0.2),
                    'criterion': 'L1Loss',
                    'num_classes': 1, # compute regression
                    # ref Original Paper
                    'batch_size': 32,
                    'learning_rate': 5e-4,
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
                'sims':{
                    'hidden_dims': (128, 16, 128),
                    'text_out': 32,
                    'post_fusion_dim': 32,
                    'dropouts': (0.2, 0.2, 0.2, 0.2),
                    'criterion': 'L1Loss',
                    'num_classes': 1, # compute regression
                    # ref Original Paper
                    'batch_size': 32,
                    'learning_rate': 5e-4,
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
            },
        }
        return tmp
   
    def __Graph_MFN(self):
        tmp = {
            'commonParas':{
                'need_align': True,
                # Tuning
                'early_stop': 20,
                # Logistics
                'weight_decay': 0.0,
            },
            # dataset
            'datasetParas':{
                'mosei':{
                    'hidden_dims': (128, 16, 128),
                    'memsize': 300,
                    'inner_node_dim': 64,
                    'output_dim': 1,
                    'NNConfig': {"drop": 0.2, "shapes": 128},
                    'gamma1Config': {"drop": 0.0, "shapes": 128},
                    'gamma2Config': {"drop": 0.5, "shapes": 128},
                    'outConfig': {"drop": 0.0, "shapes": 32},
                    'criterion': 'L1Loss',
                    # ref Original Paper
                    'batch_size': 64,
                    'learning_rate': 0.005,
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
                'mosi':{
                    'hidden_dims': (128, 16, 128),
                    'memsize': 300,
                    'inner_node_dim': 64,
                    'output_dim': 1,
                    'NNConfig': {"drop": 0.2, "shapes": 128},
                    'gamma1Config': {"drop": 0.0, "shapes": 128},
                    'gamma2Config': {"drop": 0.5, "shapes": 128},
                    'outConfig': {"drop": 0.0, "shapes": 32},
                    'criterion': 'L1Loss',
                    # ref Original Paper
                    'batch_size': 64,
                    'learning_rate': 0.005,
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
                'sims':{
                    'hidden_dims': (128, 16, 128),
                    'memsize': 64,
                    'windowsize': 2,
                    'output_dim': 1,
                    'NN1Config': {"drop": 0.2, "shapes": 32},
                    'NN2Config': {"drop": 0.7, "shapes": 128},
                    'gamma1Config': {"drop": 0.7, "shapes": 256},
                    'gamma2Config': {"drop": 0.7, "shapes": 32},
                    'outConfig': {"drop": 0.2, "shapes": 32},
                    'criterion': 'L1Loss',
                    # ref Original Paper
                    'batch_size': 128,
                    'learning_rate': 5e-4,
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                },
            },
        }
        return tmp

    def __MTFN(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'multi_label': True,
                'need_normalize': True,
                # Tuning
                'early_stop': 20,
                # Logistics
                'weight_decay': 0.0,
            },
            # dataset
            'datasetParas':{
                'sims':{
                    'hidden_dims': (256,32,256),
                    'text_out': 128,
                    'post_fusion_dim': 64,
                    'post_text_dim': 16,
                    'post_audio_dim': 4,
                    'post_video_dim': 8,
                    'dropouts': (0.3,0.3,0.3),
                    'post_dropouts': (0.4,0.4,0.4,0.4),
                    'criterion': 'L1Loss',
                    'num_classes': 1, # compute regression
                    # ref Original Paper
                    'batch_size': 64,
                    'learning_rate': 1e-3,
                    'grad_clip': 0.0, # gradient clip value (default: 0.8)
                    'M': 0.2,
                    'T': 0.4,
                    'A': 1.0,
                    'V': 0.6,
                    'text_weight_decay': 1e-5,
                    'audio_weight_decay': 1e-3,
                    'video_weight_decay': 1e-4,
                }
            },
        }
        return tmp

    def __MLMF(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'need_normalize': True,
                # Tuning
                'early_stop': 20,
            },
            # dataset
            'datasetParas':{
                'sims':{
                    'hidden_dims': (64, 16, 64),
                    'post_text_dim': 64,
                    'post_audio_dim': 5,
                    'post_video_dim': 16,
                    # dropout
                    'post_dropouts': (0.3,0.3,0.3),
                    'dropouts': (0.2, 0.2, 0.2, 0.2),
                    'output_dim': 1,
                    'criterion': 'L1Loss',
                    'rank': 4,
                    'use_softmax': False,
                    # ref Original Paper
                    'batch_size': 64,
                    'learning_rate': 0.001,
                    'factor_lr': 0.0005, # factor_learning_rate
                    'weight_decay': 0.001,
                    'grad_clip': 0.0, # gradient clip value
                    'M': 0.4,
                    'T': 0.6,
                    'A': 1.0,
                    'V': 0.2,
                    'text_weight_decay': 1e-4,
                    'audio_weight_decay': 1e-5,
                    'video_weight_decay': 1e-4, 
                },
            },
        }
        return tmp
    
    def __MLF_DNN(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'multi_label': True,
                'need_normalize': True,
                # Tuning
                'early_stop': 20,
                # Logistics
                'weight_decay': 0.0,
            },
            # dataset
            'datasetParas':{
                'sims':{
                    'hidden_dims': (128, 16, 128),
                    'text_out': 32,
                    'post_fusion_dim': 128,
                    'post_text_dim':32,
                    'post_audio_dim': 5,
                    'post_video_dim': 16,
                    'dropouts': (0.2, 0.2, 0.2),
                    'post_dropouts': (0.5,0.5,0.5,0.5),
                    'criterion': 'L1Loss',
                    'num_classes': 1, # compute regression
                    # ref Original Paper
                    'batch_size': 64,
                    'learning_rate': 0.0005,
                    'grad_clip': 0.0,
                    'M': 0.8,
                    'T': 0.6,
                    'A': 0.4,
                    'V': 0.2,
                    'text_weight_decay': 0.0,
                    'audio_weight_decay': 0.0,
                    'video_weight_decay': 0.0,
                },
            },
        }
        return tmp
    
    def __DMMSA(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'need_normalize': False,
                'use_bert': True,
                'use_finetune': True,
                'early_stop': 8,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'update_epochs': 4,
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_other': 5e-5,
                    'weight_decay_bert': 0.001,
                    'weight_decay_other': 0.01,
                    'num_classes': 1,
                    # feature subNets
                    'a_lstm_hidden_size': 64,
                    'v_lstm_hidden_size': 128,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 64,
                    'video_out': 128, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':64,
                    'post_audio_dim': 32,
                    'post_video_dim': 32,
                    'post_fusion_dropout': 0.1,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.0,
                    'post_video_dropout': 0.1,
                    'M': 1.0,
                    'T': 0.6,
                    'A': 0.6,
                    'V': 1.0,
                    # label shifting networks
                    'D': 0.1,
                    'H': 3
                },
                'mosei':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'update_epochs': 4,
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_other': 5e-5,
                    'weight_decay_bert': 0.001,
                    'weight_decay_other': 0.01,
                    'num_classes': 1,
                    # feature subNets
                    'a_lstm_hidden_size': 64,
                    'v_lstm_hidden_size': 128,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 64,
                    'video_out': 128, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':64,
                    'post_audio_dim': 32,
                    'post_video_dim': 32,
                    'post_fusion_dropout': 0.1,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.0,
                    'post_video_dropout': 0.1,
                    'M': 1.0,
                    'T': 0.6,
                    'A': 0.6,
                    'V': 1.0,
                    # label shifting networks
                    'D': 0.1,
                    'H': 3
                },
                'sims':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'update_epochs': 4,
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_other': 5e-5,
                    'weight_decay_bert': 0.001,
                    'weight_decay_other': 0.01,
                    'num_classes': 1,
                    # feature subNets
                    'a_lstm_hidden_size': 64,
                    'v_lstm_hidden_size': 128,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768,
                    'audio_out': 64,
                    'video_out': 128, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':32,
                    'post_audio_dim': 16,
                    'post_video_dim': 16,
                    'post_fusion_dropout': 0.1,
                    'post_text_dropout': 0.0,
                    'post_audio_dropout': 0.1,
                    'post_video_dropout': 0.0,
                    'M': 1.0,
                    'T': 1.0,
                    'A': 1.0,
                    'V': 0.2,
                    # label shifting networks
                    'D': 0.1,
                    'H': 3
                },
            },
        }
        return tmp
    
    def __LSMMSA(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'need_normalize': False,
                'use_bert': True,
                'use_finetune': True,
                'early_stop': 12,
                'task_type': 'regression' # regression / classification
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'update_epochs': 4,
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 1e-3,
                    'learning_rate_video': 1e-4,
                    'learning_rate_other': 1e-3,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.001,
                    'weight_decay_other': 0.001,
                    'num_classes': 1,
                    # feature subNets
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':64,
                    'post_audio_dim': 16,
                    'post_video_dim': 32,
                    'post_fusion_dropout': 0.1,
                    'post_text_dropout': 0.0,
                    'post_audio_dropout': 0.1,
                    'post_video_dropout': 0.1,
                    # res
                    'H': 3.0
                },
                'mosei':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'update_epochs': 4,
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 0.005,
                    'learning_rate_video': 1e-4,
                    'learning_rate_other': 1e-3,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.0,
                    'weight_decay_video': 0.0,
                    'weight_decay_other': 0.01,
                    'num_classes': 1,
                    # feature subNets
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 32,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':32,
                    'post_audio_dim': 16,
                    'post_video_dim': 16,
                    'post_fusion_dropout': 0.0,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.0,
                    'post_video_dropout': 0.1,
                    # res
                    'H': 3.0
                },
                'sims':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'update_epochs': 4,
                    'batch_size': 16,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 0.005,
                    'learning_rate_video': 0.005,
                    'learning_rate_other': 0.001,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.0,
                    'weight_decay_other': 0.001,
                    'num_classes': 1,
                    # feature subNets
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 64,
                    'post_text_dim':64,
                    'post_audio_dim': 16,
                    'post_video_dim': 16,
                    'post_fusion_dropout': 0.0,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.0,
                    'post_video_dropout': 0.1,
                    # res
                    'H': 1.0
                },
            },
        }
        return tmp

    def get_config(self):
        # normalize
        model_name = str.lower(self.globalArgs.modelName)
        dataset_name = str.lower(self.globalArgs.datasetName)
        # integrate all parameters
        res =  Storage(dict(vars(self.globalArgs),
                            **self.HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            **self.HYPER_MODEL_MAP[model_name]()['commonParas'],
                            **self.HYPER_DATASET_MAP[dataset_name]))
        return res