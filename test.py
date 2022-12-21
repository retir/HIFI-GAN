import os
import argparse
import collections
import warnings
import utils
import torchaudio

import numpy as np
import torch

import model as module_arch
import loss as module_loss
from tqdm import tqdm
from trainer import Trainer
from utils import prepare_device
from utils.parse_config import ConfigParser
from datasets.mel_transformer import mel_transformer
from scipy.io.wavfile import read
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

    

def main(config, args):
        
    # prepare model
    device, device_ids = prepare_device(config["n_gpu"])
    gen = config.init_obj(config["gen_arch"], module_arch)
    gen = gen.to(device)
    
    if len(device_ids) > 1:
        gen = torch.nn.DataParallel(gen, device_ids=device_ids)
        
    checkpoint = torch.load(args.checkpoint_pth, device)
    #assert checkpoint["config"]["gen_arch"] == config["gen_arch"]
    gen.load_state_dict(checkpoint["gen_state_dict"])
    gen.eval()
    gen.remove_weight_norm() # ?
    
    save_dir = args.results_dir
    os.makedirs(save_dir, exist_ok=True)
    
    base_paths = sorted(os.listdir(args.audio_pth))
    paths = [args.audio_pth + path for path in base_paths]
    for base_path, path in zip(base_paths, paths):
        sr, audio_wav = read(path)
        audio_wav = torch.FloatTensor(audio_wav)[None,:]
        if torch.max(audio_wav) >= 1e3:
            audio_wav /= 32768.0
        batch = mel_transformer(audio_wav.to(device))
        with torch.no_grad():
            batch = batch.to(device)
            gen_out = gen({'mels': batch, 'wavs': audio_wav.to(device)})
            torchaudio.save(save_dir + '/' + base_path, gen_out['g_pred'].cpu(), 22050)
            l1_loss = F.l1_loss(gen_out['mels'], gen_out['g_pred_mel']).item()
            print(f'L1 loss for {base_path} is {l1_loss}')


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        type=str,
        help="path to model config",
    )
    args.add_argument(
        "-pth",
        "--checkpoint_pth",
        type=str,
        help="path to model checkpoint",
    )
    args.add_argument(
        "-a",
        "--audio_pth",
        type=str,
        help="path to inference audio",
    )
    args.add_argument(
        "-r",
        "--results_dir",
        default="./results",
        type=str,
        help="path to results dir",
    )
    args.add_argument(
        "-device",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-z",
        "--resume",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    options = []
    config = ConfigParser.from_args(args, options)
    args = args.parse_args()
    main(config, args)