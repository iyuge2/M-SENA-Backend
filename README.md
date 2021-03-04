![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
![Torch 1.2](https://img.shields.io/badge/torch-1.2-green.svg)
![Flask 1.1.2](https://img.shields.io/badge/flask-1.1.2-green.svg)
![License](https://img.shields.io/badge/license-GPLv3-blue.svg)

This project is the backend of the [M-SENA Platform](https://github.com/thuiar/M-SENA/).

- [Installation](#installation)
  - [Docker](#docker)
  - [Source code](#source-code)
- [Reference](#reference)
  - [Datasets](#datasets)
  - [Codes](#codes)

## Installation

### Docker

See [M-SENA-frontend](https://github.com/FlameSky-S/M-SENA-frontend#docker)

### Source code

- Clone the repository

```shell
git clone https://github.com/iyuge2/M-SENA-Backend.git
cd M-SENA-Backend
```

- Install requirements
  -  Install mysql (version 5.7.32)
  -  Install python requirements
    ```
    conda create --name sena python=3.6
    source active sena
    pip install -r requirements.txt
    ```
  - Download [Bert-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) from [Google-Bert](https://github.com/google-research/bert). Then, convert Tensorflow into pytorch using [transformers-cli](https://huggingface.co/transformers/converting_tensorflow_models.html)  
  - Install [Openface Toolkits](https://github.com/TadasBaltrusaitis/OpenFace/wiki)

- Download datasets and format them using `MM-Codes/data/DataPre.py`
- Update global variables in `constants.py`
- Update basic config in `config.sh`
- Run

```
source config.sh
flask run --host=0.0.0.0
```

## Reference

### Datasets

In this section, we introduce the organizational structure of datasets, which should comply with the following structure.

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
  
We provide the download link for [preprocessed SIMS](https://pan.baidu.com/s/13Ax18SWnHRWCUJB2i8NsVw), `code: 4aa6`, `md5: 3befed5d2f6ea63a8402f5875ecb220d`, which follows the above requirements. You can get more datasets from [CMU-MultimodalSDK](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/). It's worth noting that it is not necessary to put datasets and codes together.

### Codes

- Web Interface Codes

In the backend, we use `Flask` + `Mysql` to provide the accessible requests.

```txt
.
├── AL-Codes                # Active learning codes
├── MM-Codes                # MSA algorithm codes
├── app.py                  # Flask main codes
├── config.py               # Basic config
├── config.sh               # Basic config
├── constants.py            # Global variable definition
├── database.py             # Mysql database definition
├── httpServer.py           # Dataset http server
└── requirements.txt        # Python requirements
```

- MM-Codes

> MSA Code Framework

Based on [MMSA](https://github.com/thuiar/MMSA), all model and dataset parameters are saved in `MM-Codes/config.json`.

- AL-Codes

> Labeling based  on Active Learning Code Framework

Based on [MMSA](https://github.com/thuiar/MMSA), all model and dataset parameters are saved in `AL-Codes/config.json`.
