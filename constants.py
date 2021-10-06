# HTTP CODEs
SUCCESS_CODE = 200 
WARNING_CODE = 4030 
ERROR_CODE = 4040

SQL_MAX_TEXT_LEN = 1024

# DATA-END
DATASET_ROOT_DIR = "/home/sharing/disk3/Datasets/MMSA-Standard"
DATASET_SERVER_PORT = 8096
DATASET_SERVER_IP = f"http://166.111.138.101:{DATASET_SERVER_PORT}"
MAX_ARGS_LEN = 1024 # max length of model parameters (string)
TRAIN_MODE = 0
VALID_MODE = 1
TEST_MODE = 2
LABEL_NAME_N2I = {"-": -1, "Negative": 0, "Neutral": 1, "Positive": 2}
LABEL_NAME_I2N = {-1: "-", 0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_BY_N2I = {"Unlabeled": -1, "Human": 0, "Machine": 1, "Medium": 2, "Hard": 3}
LABEL_BY_I2N = {-1: "Unlabeled", 0: "Human", 1: "Machine", 2: "Medium", 3: "Hard"}
MANUAL_LABEL_BATCH_SIZE = 16

OPENFACE_FEATURE_PATH = '/home/iyuge2/ToolKits/OpenFace/build/bin/FeatureExtraction' # TODO

# MODEL-END
MM_CODES_PATH = '/home/mhs/Projects/M-SENA-Backend/MMSA'
MODEL_TMP_SAVE = '/home/mhs/Projects/M-SENA-Backend/MMSA/results'
AL_CODES_PATH = '/home/mhs/Projects/M-SENA-Backend/ALCodes'

# ANALYSIS-END
LIVE_TMP_PATH = '/tmp/M-SENA'

# limit sequence length
MAX_TEXT_SEQ_LEN = 20
MAX_VIDEO_SEQ_LEN = 50
MAX_AUDIO_SEQ_LEN = 50

# 
LOG_FILE_PATH = './log/M-SENA.log'
APP_SETTINGS = 'config.ProductionConfig'
DATABASE_URL = 'mysql://sena:wasdwasd@127.0.0.1/sena'