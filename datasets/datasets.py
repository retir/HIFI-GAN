import torchaudio
import torch
import numpy as np
from scipy.io.wavfile import read


class BaseDataset:
    def __init__(self, data_path, split_path):
        
        with open(split_path, 'r') as f:
            all_lines = f.readlines()
        
        self.audio_pathes = [data_path + line.split('|')[0] + '.wav' for line in all_lines]
        self.buffer = []
        for path in self.audio_pathes:
            sr, audio_wav = read(path)
            audio_wav = torch.FloatTensor(audio_wav)[None,:] / 32768.0
            self.buffer.append(audio_wav)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        audio_wav = self.buffer[idx]
        if audio_wav.size(1) > 8192:
            start_ind = np.random.randint(audio_wav.size(1) - 8192)
            audio_wav = audio_wav[:,start_ind:start_ind + 8192]
        return audio_wav