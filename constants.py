# DATA-END
MAX_TEXT_LEN = 512
MAX_ARGS_LEN = 1024 # max length of model parameters (string)
TRAIN_MODE = 0
VALID_MODE = 1
TEST_MODE = 2
SUPPORT_FORMAT = {
    'text_format': ['csv'],
    'audio_format': ['wav', 'mp3'],
    'video_format': ['mp4', 'avi', 'flv'],
}

# MODEL-END
CODE_PATH = 'M-Codes'
MODEL_SAVE_PATH = 'M-Codes/model_saves'

# HTTP CODEs
SUCCESS_CODE = 200 
WARNING_CODE = 4030 
ERROR_CODE = 4040
