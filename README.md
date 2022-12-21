# HIFI-GAN project

## Installation guide

First of all you need to download the repository: 

```shell
git clone https://github.com/retir/HIFI-GAN
cd HIFI-GAN
```
Then install necessary dependencies:

```shell
pip install -r requirements.txt
```

If you need, wonload pretrained model and train\val split for LJSpeech

```shell
python3 downloader.py
```

If you want to inference pretrained model on wav's use
```shell
python3 test.py -c path/to/config.json  -pth path/to/checkpoint.pth -r path/to/result_dir -a path/to/audio_for_eval/
```

Model from `downloader.py` was trained on config `configs/hifigan_v1.json`

If you want to train your own model use

```shell
python3 train.py -c path/to/config.json
```

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.