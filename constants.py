# HTTP CODEs
SUCCESS_CODE = 200 
WARNING_CODE = 4030 
ERROR_CODE = 4040

# DATA-END
DATASET_ROOT_DIR = "/home/iyuge2/Project/M-SENA/Datasets"
DATASET_SERVER_PORT = 8096
DATASET_SERVER_IP = f"http://166.111.138.149:{DATASET_SERVER_PORT}"
MAX_ARGS_LEN = 1024 # max length of model parameters (string)
TRAIN_MODE = 0
VALID_MODE = 1
TEST_MODE = 2
LABEL_NAME_N2I = {"Neutral": 0, "Negative": 1, "Weakly Negative": 2, "Positive": 3, "Weakly Positive": 4}
LABEL_NAME_I2N = {0: "Neutral", 1: "Negative", 2: "Weakly Negative", 3: "Positive", 4: "Weakly Positive"}
LABEL_BY_N2I = {"Unlabeled": -1, "Human": 0, "Machine": 1, "Middle": 2, "Hard": 3}
LABEL_BY_I2N = {-1: "Unlabeled", 0: "Human", 1: "Machine", 2: "Middle", 3: "Hard"}
MANUAL_LABEL_BATCH_SIZE = 2

# MODEL-END
MM_CODES_PATH = 'M-Codes'
MODEL_TMP_SAVE = 'M-Codes/results/model_tmp_saves'
MODEL_FINAL_SAVE = 'M-Codes/results/model_final_saves'
AL_CODES_PATH = 'AL-Codes'
# limit sequence length
MAX_TEXT_SEQ_LEN = 20
MAX_VIDEO_SEQ_LEN = 50
MAX_AUDIO_SEQ_LEN = 50
