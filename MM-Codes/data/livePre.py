import os
import sys
import torch
import librosa
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from aip import AipSpeech
from facenet_pytorch import MTCNN, InceptionResnetV1
from transformers import *

from constants import *

class MLive():
    def __init__(self, working_dir, transcript, language="Chinese"):
        self.working_dir = working_dir
        
        self.video_path = os.path.join(self.working_dir, 'live.mp4')
        self.audio_path = os.path.join(self.working_dir, 'live.wav')

        self.text = transcript

        self.language = language

        self.faces_feature_dir = os.path.join(self.working_dir, 'OpenFace')

        # self.frames_dir = os.path.join(self.working_dir, "Frames")
        # if not os.path.exists(self.frames_dir):
        #     os.mkdir(self.frames_dir)

        # self.faces_dir = self.frames_dir.replace("Frames", "Faces")
        # if not os.path.exists(self.faces_dir):
        #     os.mkdir(self.faces_dir)

        # Embeddings
        if language == "Chinese":
            self.pretrainedBertPath = os.path.join(MM_CODES_PATH, 'pretrained_model/bert_cn')
        else:
            self.pretrainedBertPath = os.path.join(MM_CODES_PATH, 'pretrained_model/bert_en')

    def __FetchAudio(self):
        if not os.path.exists(self.audio_path):
            # 调用ffmpeg执行音频提取功能
            cmd = 'ffmpeg -i ' + self.video_path + ' -f wav -vn ' + \
                self.audio_path + ' -loglevel quiet'
            os.system(cmd)

    def __FetchText(self):
        def get_file_content(filePath):
            with open(filePath, 'rb') as fp:
                return fp.read()

        client = AipSpeech(self.APP_ID, self.API_KEY, self.SECRET_KEY)
        result = client.asr(get_file_content(self.audio_path), 'wav', 16000, 
                            {'dev_pid': self.DEV_PID})
        if 'err_msg' in result.keys() and result['err_msg'] == "success.":
            text = result['result'][0]
        else:
            text = 'ASR ERROR!'
        with open(self.text_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    def __FetchFrames(self):
        cmd = "ffmpeg -i " + self.video_path + " -r 30 " + self.frames_dir + "/%04d.png -loglevel quiet"
        os.system(cmd)
    
    def __AlignFaces(self):
        mtcnn = MTCNN(image_size=224, margin=0)
        frames_pathes = sorted(glob(os.path.join(self.frames_dir, "*.png")))
        
        for frames_path in frames_pathes:
            output_path = frames_path.replace("Frames", "Faces")
            try:
                img = Image.open(frames_path)
                mtcnn(img, save_path=output_path)
            except Exception as e:
                continue
    
    def __FetchVideo(self):
        if not os.path.exists(self.faces_feature_dir):
            cmd = OPENFACE_FEATURE_PATH + ' -f ' + self.video_path + ' -out_dir ' + self.faces_feature_dir
            print(cmd)
            os.system(cmd)

    def __getVideoEmbedding(self, pool_size=5):
        df = pd.read_csv(os.path.join(self.faces_feature_dir, 'live.csv'))
        features, local_features = [], []
        for i in range(len(df)):
            local_features.append(np.array(df.loc[i][df.columns[5:]]))
            if (i + 1) % pool_size == 0:
                features.append(np.array(local_features).mean(axis=0))
                local_features = []
        if len(local_features) != 0:
            features.append(np.array(local_features).mean(axis=0))
        return np.array(features)

    def __getAudioEmbedding(self):
        y, sr = librosa.load(self.audio_path)
        # using librosa to get audio features (f0, mfcc, cqt)
        hop_length = 512 # hop_length smaller, seq_len larger
        f0 = librosa.feature.zero_crossing_rate(y, hop_length=hop_length).T # (seq_len, 1)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, htk=True).T # (seq_len, 20)
        cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length).T # (seq_len, 12)

        return np.concatenate([f0, mfcc, cqt], axis=-1)

    def __getTextEmbedding(self):
        # with open(self.text_path, 'r', encoding='utf-8') as fp:
        #     text = fp.read().strip()
        tokenizer_class = BertTokenizer
        model_class = BertModel
        # directory is fine
        pretrained_weights = self.pretrainedBertPath
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        # add_special_tokens will add start and end token
        input_ids = torch.tensor([tokenizer.encode(self.text, add_special_tokens=True)])
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze().numpy()
    
    def dataPre(self):
        print("fetch audio...")
        self.__FetchAudio()
        # print("fetch text...")
        # self.__FetchText()
        # print("fetch frames...")
        # self.__FetchFrames()
        # print("align faces...")
        # self.__AlignFaces()
        print("extract video features...")
        self.__FetchVideo()
   
    def getEmbeddings(self, seq_lens, feature_dims):
        def seq_len_padding(x, dst_len):
            if x.shape[0] < dst_len:
                x = np.concatenate([x, np.zeros([dst_len - x.shape[0], x.shape[1]])], axis=0)
            else:
                x = x[:dst_len]
            return x
        
        def feature_dim_padding(x, dst_len):
            if x.shape[1] < dst_len:
                x = np.concatenate([x, np.random.random([x.shape[0], dst_len - x.shape[1]])], axis=1)
            else:
                x = x[:,:dst_len]
            return x
 
        text_e = self.__getTextEmbedding()
        audio_e = self.__getAudioEmbedding()
        video_e = self.__getVideoEmbedding()
        
        # padding seqs
        text_e = seq_len_padding(text_e, seq_lens[0])
        audio_e = seq_len_padding(audio_e, seq_lens[1])
        video_e = seq_len_padding(video_e, seq_lens[2])
        # padding feature_dims
        text_e = feature_dim_padding(text_e, feature_dims[0])
        audio_e = feature_dim_padding(audio_e, feature_dims[1])
        video_e = feature_dim_padding(video_e, feature_dims[2])

        return text_e, audio_e, video_e
    
    def clash(self):
        cmd = 'rm -rf ' + self.working_dir
        os.system(cmd)

if __name__ == "__main__":
    op = MLive('/home/iyuge2/Project/M-SENA/Datasets/tmp/0001/0001.mp4')
    op.dataPre()
    # op.getEmbeddings()
