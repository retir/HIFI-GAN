import torch
import torchaudio
import librosa 
import torch.nn.functional as F

from torch import nn
from dataclasses import dataclass
from datasets.mel_transformer import mel_transformer


def collate_func(batch):
    max_audio_len = 0
    for audio in batch:
        if audio.size(1) > max_audio_len:
            max_audio_len = audio.size(1)
    result_batch = []
    for audio in batch:
        result_batch.append(F.pad(audio[0], (0, max_audio_len - audio.size(1)), value=0.))
    result_batch = torch.stack(result_batch, dim=0)
    return {'wavs': result_batch}