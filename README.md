![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
![Torch 1.2](https://img.shields.io/badge/torch-1.2-green.svg)
![Flask 1.1.2](https://img.shields.io/badge/flask-1.1.2-green.svg)
![License](https://img.shields.io/badge/license-GPLv3-blue.svg)

This project is the backend of the [M-SENA Platform](https://github.com/thuiar/M-SENA/).

- [Installation](#installation)
  - [Docker](#docker)
  - [From Source](#from-source)
- [Reference](#reference)
  - [Dataset Structure](#dataset-structure)
  - [Code Structure](#code-structure)

# Installation

## Docker

We provide a [docker image](https://hub.docker.com/r/flamesky/m-sena-platform/) of our platform. See the [main repo](https://github.com/thuiar/M-SENA#docker) for instructions. 

## From Source

### 1. Clone this Repository

```shell
$ git clone https://github.com/iyuge2/M-SENA-Backend.git
$ cd M-SENA-Backend
```

### 2. Install Requirements

  - Install system requirements

  ```
  $ apt install mysql-server default-libmysqlclient-dev libsndfile1 ffmpeg
  ```
  
  - Install python requirements

  ```
  $ conda create --name sena python=3.8
  $ source active sena
  $ pip install -r requirements.txt
  ```
    
  - Download [Bert-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) from [Google-Bert](https://github.com/google-research/bert). Then, convert Tensorflow into pytorch using [transformers-cli](https://huggingface.co/transformers/converting_tensorflow_models.html). Place the converted model under `MM-Codes/pretrained_model` directory.  
  - Install [Openface Toolkits](https://github.com/TadasBaltrusaitis/OpenFace/wiki)

### 3. Configure MySQL

  - Login MySQL with root

  ```
  $ mysql -u root -p
  ```

  - Create a database for M-SENA

  ```
  mysql> CREATE DATABASE sena;
  ```
  
  - Create a user for M-SENA and grant privileges

  ```
  mysql> CREATE USER sena IDENTIFIED BY 'MyPassword';
  mysql> GRANT ALL PRIVILEGES ON sena.* TO sena@`%`;
  mysql> FLUSH PRIVILEGES;
  ```

### 4. Configs 

  - Edit `Constants.py`. Alter `DATASET_ROOT_DIR`, `DATASET_SERVER_IP`, `OPENFACE_FEATURE_PATH`, `MM_CODES_PATH`, `MODEL_TMP_SAVE`, `AL_CODES_PATH` and `LIVE_TMP_PATH` to fit your settings. 
  - Edit `config.sh`. Look for `DATABASE_URL` and change it to fit your database settings.

### 5. Datasets 

  - Download datasets and locate them under `DATASET_ROOT_DIR` specified in `constants.py`
  - Add information in `DATASET_ROOT_DIR/config.json` file to register the new dataset. 
  - Format datasets with `MM-Codes/data/DataPre.py`
  - For datasets that needs labeling, the config file locates in `AL-Codes` directory. 
  
  ```
  $ python MM-Codes/data/DataPre.py --working_dir $PATH_TO_DATASET --openface2Path $PATH_TO_OPENFACE2_FeatureExtraction_TOOL --language cn/en
  ```
  
  - The structure of the `DATASET_ROOT_DIR` directory is introduced in the [next](#datasets) section. 

### 6. Run

```
$ source config.sh
$ flask run --host=0.0.0.0
```

# Reference

## Dataset Structure

The structure of the root dataset directory should look like this:

```txt
.
├── config.json
├── MOSEI
│   ├── label.csv
│   ├── Processed
│   └── Raw
├── MOSI
│   ├── label.csv
│   ├── Processed
│   └── Raw
└── SIMS
    ├── label.csv
    ├── Processed
    └── Raw
```

- `config.json`: stating necessary information for all datasets. For example, `language`, `label_path`, `features`, etc. It only works when scanning and updating datasets.
- `**/label.csv`: storing detailed information for each video clip in `**` dataset, including `video_id`, `clip_id`, `normal text`, `label value (Float)`, `annotation (String)`, `mode (training attributes)`. Besides, we define a field `label_by` to indicate the label type, which is necessary for labeling based on active learning.

![dataset-Label](assets/dataset-label.png)

- `**/Processed`: placing feature files. We use `pickle` to store processed features, which are organized as the following structure. These files are used in `MM-Codes`.

```python
{
    "train": {
        "raw_text": [],
        "audio": [],
        "vision": [],
        "id": [], # [video_id$_$clip_id, ..., ...]
        "text": [],
        "text_bert": [],
        "audio_lengths": [],
        "vision_lengths": [],
        "annotations": [],
        "classification_labels": [], # Negative(< 0), Neutral(0), Positive(> 0)
        "regression_labels": []
    },
    "valid": {***}, # same as the "train"
    "test": {***}, # same as the "train"
}
```

- `**/Raw`: placing raw videos. The path of each clip should be consistent with `label.csv`.
  
We provide the download link for [preprocessed SIMS](https://pan.baidu.com/s/13Ax18SWnHRWCUJB2i8NsVw), `code: 4aa6`, `md5: 3befed5d2f6ea63a8402f5875ecb220d`, which follows the above requirements. You can get more datasets from [CMU-MultimodalSDK](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/). 

## Code Structure

The source code is organized as follows: 

```txt
.
├── AL-Codes                # Active learning codes
├── MM-Codes                # MSA algorithm codes
├── app.py                  # Flask main codes
├── config.py               # Basic config
├── config.sh               # Basic config
├── constants.py            # Global variable definition
├── database.py             # Database definition & initialization
├── httpServer.py           # Dataset server (for video previews)
└── requirements.txt        # Python requirements
```

- MM-Codes

> MSA Code Framework

Based on [MMSA](https://github.com/thuiar/MMSA), all model and dataset parameters are saved in `MM-Codes/config.json`.

- AL-Codes

> Labeling based  on Active Learning Code Framework

Based on [MMSA](https://github.com/thuiar/MMSA), all model and dataset parameters are saved in `AL-Codes/config.json`.
