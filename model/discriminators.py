import torch
import torch.nn.functional as F
import torch.nn as nn
import model.discriminators as disc_module
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils.model_utils import init_weights, get_padding


class PeriodDisc(torch.nn.Module):
    def __init__(self, period):
        super(PeriodDisc, self).__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(Conv2d(1, 32, (5, 1), (3, 1), padding=(get_padding(5, 1), 0))),
            weight_norm(Conv2d(32, 128, (5, 1), (3, 1), padding=(get_padding(5, 1), 0))),
            weight_norm(Conv2d(128, 512, (5, 1), (3, 1), padding=(get_padding(5, 1), 0))),
            weight_norm(Conv2d(512, 1024, (5, 1), (3, 1), padding=(get_padding(5, 1), 0))),
            weight_norm(Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = weight_norm(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        x_len = x.size(2)
        tail = x_len % self.period
        if tail > 0:
            x = F.pad(x, (0, self.period - tail), "reflect")
            x_len = x_len - tail + self.period
        x = x.view(x.size(0), x.size(1), x_len // self.period, self.period)

        features = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        x = torch.flatten(x, 1, -1)

        return x, features


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = []
        for period in [2, 3, 5, 7, 11]:
            self.discriminators.append(PeriodDisc(period))
        self.discriminators = nn.ModuleList(self.discriminators)

    def forward(self, y, y_hat):
        real_preds = []
        fake_preds = []
        real_feats = []
        fake_feats = []
        for i, disc in enumerate(self.discriminators):
            real_pred, real_feat = disc(y)
            real_preds.append(real_pred)
            real_feats.append(real_feat)
            
            fake_pred, fake_feat = disc(y_hat)
            fake_preds.append(fake_pred)
            fake_feats.append(fake_feat)

        return real_preds, fake_preds, real_feats, fake_feats


class ScaleDisc(torch.nn.Module):
    def __init__(self, norm=weight_norm):
        super(ScaleDisc, self).__init__()
        self.convs = nn.ModuleList([
            norm(Conv1d(1, 128, 15, 1, padding=7)),
            norm(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        features = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        x = torch.flatten(x, 1, -1)

        return x, features


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            ScaleDisc(spectral_norm),
            ScaleDisc(),
            ScaleDisc(),
        ])
        self.pools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        real_preds = []
        fake_preds = []
        real_feats = []
        fake_feats = []
        for i, disc in enumerate(self.discriminators):
            if i != 0:
                y = self.pools[i-1](y)
                y_hat = self.pools[i-1](y_hat)
            real_pred, real_feat = disc(y)
            real_preds.append(real_pred)
            real_feats.append(real_feat)
            
            fake_pred, fake_feat = disc(y)
            fake_preds.append(fake_pred)
            fake_feats.append(fake_feat)

        return real_preds, fake_preds, real_feats, fake_feats
    

class Multidisc(nn.Module):
    def __init__(self, mpd_args, msd_args):
        super(Multidisc, self).__init__()
        self.mpd = MultiPeriodDiscriminator(**mpd_args)
        self.msd = MultiScaleDiscriminator(**msd_args)

    
    def forward(self, batch, step='D'):
        y = batch['wavs'].unsqueeze(1)
        if step == 'D':
            pred = batch['g_pred'].detach().unsqueeze(1)
            real_preds_mpd, fake_preds_mpd, _, _ = self.mpd(y, pred)
            real_preds_msd, fake_preds_msd, _, _ = self.msd(y, pred)
            batch.update({'real_preds_mpd': real_preds_mpd, 
                          'fake_preds_mpd': fake_preds_mpd,
                          'real_preds_msd': real_preds_msd,
                          'fake_preds_msd': fake_preds_msd})
        else:
            pred = batch['g_pred'].unsqueeze(1)
            real_preds_mpd, fake_preds_mpd, real_feats_mpd, fake_feats_mpd = self.mpd(y, pred) 
            real_preds_msd, fake_preds_msd, real_feats_msd, fake_feats_msd = self.msd(y, pred)
            batch.update({'real_preds_mpd': real_preds_mpd, 
                          'fake_preds_mpd': fake_preds_mpd,
                          'real_feats_mpd': real_feats_mpd, 
                          'fake_feats_mpd': fake_feats_mpd,
                          'real_preds_msd': real_preds_msd,
                          'fake_preds_msd': fake_preds_msd,
                          'real_feats_msd': real_feats_msd,
                          'fake_feats_msd': fake_feats_msd})
        return batch
            
            
        
    