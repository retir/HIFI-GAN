import os
import re
import torch
import utils
import torchaudio
import numpy as np
from scipy.io.wavfile import read

from torch import nn
from tqdm import tqdm
from datasets.collate_fn import mel_transformer
import torch.nn.functional as F

def toggle_grad(model, mode):
    for param in model.parameters():
        param.requires_grad = mode
    return model

class Trainer:
    def __init__(self, generator, 
                 discriminator, 
                 criterion, 
                 gen_optimizer, 
                 dis_optimizer, 
                 logger, 
                 config, 
                 device, 
                 dataloaders, 
                 len_epoch,
                 mel_transformer,
                 gen_scheduler=None,
                 dis_scheduler=None,
                 skip_oom=True,):
        self.device = device
        self.config = config
        self.skip_oom = skip_oom
        
        self.gen = generator
        self.dis = discriminator
        self.criterion = criterion
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        
        self.train_loader, self.val_loader = dataloaders
        self.mel_transformer = mel_transformer
        self.len_epoch = len_epoch
        self.gen_scheduler = gen_scheduler
        self.dis_scheduler = dis_scheduler
        self.logger = logger
        
        self._last_epoch = 0
        self.current_step = 0
        
        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        
        self.start_epoch = 1
        
        self.checkpoint_dir = config.save_dir
        if config.resume is not None:
            self._resume_checkpoint(config.resume)
        
        spec_dir = config['validation']['special_path']
        self.special_paths = [spec_dir + path for path in sorted(os.listdir(spec_dir))]
       
    
    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints
        :param epoch: current epoch number
        """
        gen_arch = type(self.gen).__name__
        dis_arch = type(self.dis).__name__
        state = {
            "gen_arch": gen_arch,
            "dis_arch": dis_arch,
            "epoch": epoch,
            "step" : self.current_step,
            "gen_state_dict": self.gen.state_dict(),
            "dis_state_dict": self.dis.state_dict(),
            "gen_optimizer": self.gen_optimizer.state_dict(),
            "dis_optimizer": self.dis_optimizer.state_dict(),
            "gen_scheduler": self.gen_scheduler.state_dict(),
            "dis_scheduler": self.dis_scheduler.state_dict(),
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
            
    
    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.current_step = checkpoint["step"] + 1

        # load architecture params from checkpoint.
        if checkpoint["config"]["gen_arch"] != self.config["gen_arch"] or checkpoint["config"]["dis_arch"] != self.config["dis_arch"]:
            print(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.gen.load_state_dict(checkpoint["gen_state_dict"])
        self.dis.load_state_dict(checkpoint["dis_state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
                checkpoint["config"]["gen_optimizer"] != self.config["gen_optimizer"] or
                checkpoint["config"]["dis_optimizer"] != self.config["dis_optimizer"] or
                checkpoint["config"]["gen_scheduler"] != self.config["gen_scheduler"] or
                checkpoint["config"]["dis_scheduler"] != self.config["dis_scheduler"]
          ):
            print(
                "Warning: Optimizer or lr_scheduler given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        else:
            self.gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])
            self.dis_optimizer.load_state_dict(checkpoint["dis_optimizer"])
            self.gen_scheduler.load_state_dict(checkpoint["gen_scheduler"])
            self.dis_scheduler.load_state_dict(checkpoint["dis_scheduler"])

        print(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
    
    def validate(self):
        print('Start validating')
        self.gen.eval()
        self.logger.set_step(self.current_step, mode='test')
        l1_error = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                batch = self.prepare_batch(batch)
                batch['mels'] = mel_transformer(batch['wavs'])
                gen_out = self.gen(batch)
                #padded_mels = F.pad(gen_out['mels'], (0, gen_out['y_g_hat_mel'].size(-1) - gen_out['mels'].size(-1), 0, 0, 0, 0), mode='constant', value=-11.5129251)
                l1_error += F.l1_loss(gen_out['mels'], gen_out['y_g_hat_mel']).item()

        l1_error /= len(self.val_loader)
        self.logger.add_scalar('l1_mel_loss', l1_error)
        
        
        for i, path in enumerate(self.special_paths):
            sr, audio_wav = read(path)
            audio_wav = torch.FloatTensor(audio_wav)[None,:]
            #print('VAL', torch.max(audio_wav), torch.min(audio_wav))
            #audio_wav, sr = torchaudio.load(path)
            batch = mel_transformer(audio_wav.cuda())
            with torch.no_grad():
                batch = batch.cuda()
                #batch = self.prepare_batch(batch)
                gen_out = self.gen({'mels': batch, 'wavs': audio_wav.cuda()})
                #print(gen_out.shape)
                #start_ind = 
                self.logger.add_audio(f'val_{i}', gen_out['y_g_hat'].cpu()[0], 22050)
                #print(gen_out['wavs'].shape)
                self.logger.add_audio(f'val_{i}_true', gen_out['wavs'].cpu()[0], 22050)
                #print('OUT1', torch.max(gen_out['wavs'].cpu()[0]), torch.min(gen_out['wavs'].cpu()[0]))
                #print('OUT2', torch.max(gen_out['y_g_hat'].cpu()[0]), torch.min(gen_out['y_g_hat'].cpu()[0]))
                #print(audio_wav.cpu()[0].shape)
            
#         save_dir = str(self.checkpoint_dir) + "/results_" + str(self.current_step).zfill(7)
#         os.makedirs(save_dir, exist_ok=True)
        
        print('End validating')
        self.gen.train()
        self.logger.set_step(self.current_step)
    
    
    def prepare_batch(self, batch):
        for k, v in batch.items():
            if type(v) == torch.Tensor:
                batch[k] = v.float().to(self.device)
        return batch
        
    
    def train(self):
        tqdm_bar = tqdm(total=self.config['trainer']['epochs'] * len(self.train_loader) - self.current_step)
        self.gen.train()
        self.dis.train()
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            for i, batch in enumerate(self.train_loader):
                self.current_step += 1
                tqdm_bar.update(1)
                self.logger.set_step(self.current_step)

                # Get Data
                batch = self.prepare_batch(batch)
                batch['mels'] = mel_transformer(batch['wavs'])
                gen_out = self.gen(batch)
                
                # D-step
                toggle_grad(self.dis, True)
                self.dis_optimizer.zero_grad()
                dis_out = self.dis(gen_out, step='D')
                d_losses = self.criterion(dis_out, step='D')
                total_D_loss = sum(d_losses.values())
                
                for loss_name, loss_val in d_losses.items():
                    loss_numpy = loss_val.detach().cpu().numpy()
                    self.logger.add_scalar(loss_name, loss_numpy)
                
                total_D_loss.backward()
                self.dis_optimizer.step()
                self.logger.add_scalar('total_D_loss', total_D_loss.item())
                
                if self.current_step % 500 == 0:
                    audio = gen_out['y_g_hat'][0].detach().cpu().reshape(-1)
                    #print('TRAIN2', torch.max(audio), torch.min(audio))
                    self.logger.add_audio(f'train_example', audio , 22050)
                    audio = gen_out['wavs'][1].detach().cpu().reshape(-1)
                    #print('TRAIN', torch.max(audio), torch.min(audio))
                    #print(audio.shape)
                    self.logger.add_audio(f'train_example_true', audio, 22050)
                
                
                # G-step
                toggle_grad(self.dis, False)
                self.gen_optimizer.zero_grad()
                dis_out = self.dis(gen_out, step='G')
                g_losses = self.criterion(dis_out, step='G')
                total_G_loss = sum(g_losses.values())
                
                for loss_name, loss_val in g_losses.items():
                    loss_numpy = loss_val.detach().cpu().numpy()
                    self.logger.add_scalar(loss_name, loss_numpy)
                    
                total_G_loss.backward()
                self.gen_optimizer.step()
                self.logger.add_scalar('total_G_loss', total_G_loss.item())
                #self.validate()
                #assert 1 == 2

#                 # Clipping gradients to avoid gradient explosion
#                 nn.utils.clip_grad_norm_(
#                     self.model.parameters(), self.config['trainer']['grad_norm_clip'])

            self.gen_scheduler.step()
            self.dis_scheduler.step()
            last_lr = self.gen_scheduler.get_last_lr()[-1]
            self.logger.add_scalar("gen_lr", last_lr)
            last_lr = self.dis_scheduler.get_last_lr()[-1]
            self.logger.add_scalar("dis_lr", last_lr)
                    
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)
            if epoch % self.config['validation']['val_step'] == 0:
                self.validate()