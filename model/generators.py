import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils.model_utils import init_weights, get_padding
from datasets.mel_transformer import mel_transformer


class ResBlock(torch.nn.Module):
    def __init__(self, inner_channels, kernel_size, dilation):
        super(ResBlock, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(inner_channels, inner_channels, kernel_size, padding=get_padding(kernel_size, dilation[0]), dilation=dilation[0])),
            weight_norm(Conv1d(inner_channels, inner_channels, kernel_size, padding=get_padding(kernel_size, dilation[1]), dilation=dilation[1]))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for conv in self.convs:
            skip = x
            x = F.leaky_relu(x, 0.1)
            x = conv(x)
            x = x + skip
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, upsample_rates, upsample_kernel_sizes, upsample_initial_channel, resblock_kernel_sizes, resblock_dilation_sizes):
        super(Generator, self).__init__()
        self.mel_transformer = mel_transformer
        self.num_kernels = len(resblock_kernel_sizes)
        self.conv_pre = weight_norm(Conv1d(80, upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (stride, kern_sz) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(upsample_initial_channel // (2**i),
                                                        upsample_initial_channel // (2**(i+1)), 
                                                        kern_sz, 
                                                        stride=stride, 
                                                        padding=(kern_sz-stride) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            inner_ch = upsample_initial_channel // (2**(i+1))
            for j, (kern_sz, dilat) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(inner_ch, kern_sz, dilat))

        self.conv_post = weight_norm(Conv1d(inner_ch, 1, kernel_size=7, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, batch):
        x = batch['mels']
        x = self.conv_pre(x)
        for i, upper in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = upper(x)
            start = i * self.num_kernels
            xs = self.resblocks[start](x)
            for resblock in self.resblocks[start + 1: start + self.num_kernels]:
                xs += resblock(x) 
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        mel = self.mel_transformer(x)
        batch.update({'g_pred': x.squeeze(1), 'g_pred_mel': mel}) 
        batch['mels'] = F.pad(batch['mels'], (0, batch['g_pred_mel'].size(-1) - batch['mels'].size(-1), 0, 0), mode='constant', value=-11.5129251).unsqueeze(1)
        if 'wavs' in batch.keys():
            batch['wavs'] = F.pad(batch['wavs'], (0, batch['g_pred'].size(-1) - batch['wavs'].size(-1), 0, 0), mode='constant', value=0.)
            assert np.all(batch['mels'].shape == batch['g_pred_mel'].shape), (batch['mels'].shape, batch['g_pred_mel'].shape)
            assert np.all(batch['wavs'].shape == batch['g_pred'].shape), (batch['wavs'].shape, batch['g_pred'].shape)
        return batch
    
    def remove_weight_norm(self):
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)