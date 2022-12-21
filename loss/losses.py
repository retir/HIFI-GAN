import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class HIFIGANLoss(nn.Module):
    def __init__(self):
        super(HIFIGANLoss, self).__init__()
    
    
    def forward(self, batch, step):
        if step == 'D':
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = self.discriminator_loss(batch['y_df_hat_r'], batch['y_df_hat_g'])
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = self.discriminator_loss(batch['y_ds_hat_r'], batch['y_ds_hat_g'])
            return {'loss_disc_f': loss_disc_f, 'loss_disc_s': loss_disc_s}
        else:
            #padded_mels = F.pad(batch['mels'], (0, batch['y_g_hat_mel'].size(-1) - batch['mels'].size(-1), 0, 0, 0, 0), mode='constant', value=-11.5129251)
            loss_mel = F.l1_loss(batch['mels'], batch['y_g_hat_mel']) * 45
            loss_fm_f = self.feature_loss(batch['fmap_f_r'], batch['fmap_f_g'])
            loss_fm_s = self.feature_loss(batch['fmap_s_r'], batch['fmap_s_g'])
            loss_gen_f, losses_gen_f = self.generator_loss(batch['y_df_hat_g'])
            loss_gen_s, losses_gen_s = self.generator_loss(batch['y_ds_hat_g'])
            #loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            return {'loss_gen_s': loss_gen_s, 'loss_gen_f': loss_gen_f, 'loss_fm_s': loss_fm_s, 'loss_fm_f': loss_fm_f, 'loss_mel': loss_mel}

            
    def feature_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss*2
    
    def discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1-dr)**2)
            g_loss = torch.mean(dg**2)
            loss += (r_loss + g_loss)
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses
    
    
    def generator_loss(self, disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1-dg)**2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses
    