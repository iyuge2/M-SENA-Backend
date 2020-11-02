import requests

dataset_info = {
    # 'name', 'path', 'language', 'label_type', 'data_params', 'text_format', \
    #             'audio_format', 'video_format', 'has_s_label', 'has_feature', 'is_locked', 'description
    "name": "MOSI",
    "path": "/path/to/dataset",
    "language": "en",
    "label_type": 0,
    "data_params": "",
    "text_format": 'txt',
    "audio_format": 'wav',
    "video_format": 'mp4',
    "has_s_label": True,
    "has_feature": True,
    "is_locked": False,
    "description": ""
}

r = requests.post("http://127.0.0.1:5000/data/create_dataset", data=dataset_info)

print(r.text)