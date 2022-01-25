

# SYSTEM-WIDE
SERVER_PORT = 8000
APP_SETTINGS = 'config.ProductionConfig'
DATABASE_URL = 'mysql://sena:wasdwasd@127.0.0.1/sena'

LOG_FILE_PATH = './log/M-SENA.log'
LIVE_TMP_PATH = '/tmp/M-SENA'
MODEL_SAVE_PATH = '/home/sharing/disk3/M-SENA/saved_models'
RES_SAVE_PATH  = '/home/sharing/disk3/M-SENA/results'

# HTTP CODEs
SUCCESS_CODE = 200
ERROR_CODE = 4040

# DATA-END
DATASET_ROOT_DIR = "/home/sharing/disk3/Datasets/MMSA-Standard"
DATASET_SERVER_PORT = 8096
DATASET_SERVER_IP = f"http://166.111.138.142:{DATASET_SERVER_PORT}"
LIVE_DEMO_PORT = 8097
LIVE_DEMO_SERVER_IP = f"http://166.111.138.142:{LIVE_DEMO_PORT}"
MAX_ARGS_LEN = 2048 # max len of args column in Result table
SQL_MAX_TEXT_LEN = 1024 # max len of text column in Dsample table


# limit sequence length
# MAX_TEXT_SEQ_LEN = 20
# MAX_VIDEO_SEQ_LEN = 50
# MAX_AUDIO_SEQ_LEN = 50
