import os
import time
import random
import hashlib
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import clear_output

import torch
import torchaudio
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from metrics import AUC_fast
from models import BirdModel
from dataset import XCDataset
from losses import CASL, ASL, BCE
from spectrogram import Spectrogram
from utils import Lab, set_seed, error_analysis
from augmentation import Masking, Stretch, MixupAudio


CFG = {
    'seed':2,
    "batch_size":32, 
    "num_workers":8,
    "train_only":False,
    "lr":5e-4,
    "loss":nn.BCEWithLogitsLoss(),
    'backbone_pooling':'avg', 
    'backbone':'tf_efficientnetv2_b0',
    'dropout':.2,
    'verbose':2,
    'mel':{'n_mels':256, 'f_min':20, 'n_fft':2048, 'target_size':(224,224), 'mel_scale':'slaney', 'norm':'slaney'},
    'metrics':[AUC_fast],
    'scheduler':True,
    'model':BirdModel,
    'num_labels':234
}


def create_xc_dataloaders(config={'batch_size':32, 'num_workers':0}, fold=0):
    train_dataset = XCDataset(is_train=True, config=config, fold=fold)
    val_dataset = XCDataset(is_train=False, config=config, fold=fold)

    torch.manual_seed(config['seed'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    targets = {"labels":train_dataset.LABELS}
    
    return train_loader, val_loader, targets




class Trainer:
    
    def __init__(self, config={}, fold=0, epochs=16):
        self.config = CFG.copy()
        self.config.update(config)
        set_seed(self.config['seed'])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.exp_id = hashlib.sha256(str(time.time()).encode()).hexdigest()

        cols = ['id', 'epoch', 'train_loss', 'val_loss', 'lr', 'timestamp', 'fold']+['val_'+m.__name__ for m in self.config['metrics']]+list(self.config.keys())
        self.history = pd.DataFrame([], columns=cols)

        self.batch_size = self.config['batch_size']
        self.fold = fold

        self.audio_aug = torchvision.transforms.Compose([
        ])

        self.aug = torchvision.transforms.Compose([
            Stretch(.1, axis='x'),
            Masking(N=8, max_freq_band=2, max_time_band=3, shape='cross')
        ])

        
        self.mel = Spectrogram(**config['mel']).to(self.device)

        

    def train_one_epoch(self, epoch=0):
        self.model.train()        
        Loss = 0
        n_steps = len(self.train_loader)
        batch_size = self.config['batch_size']

        lam = epoch/self.epochs
        
        if self.config['verbose']==2: pbar = tqdm(enumerate(self.train_loader), total=n_steps, desc="Training")
        else: pbar = enumerate(self.train_loader)

        mix_audio = MixupAudio(**self.config['mix'])
        
        for batch_idx, (x, y) in pbar:
            self.optimizer.zero_grad()
            
            x = x.to(self.device)
            y = y.to(self.device)

            x_single = self.aug(self.mel(x))
            logits_single = self.model(x_single)

            x_mix, (y_mix,logits_single_mix) = mix_audio(x, [y,logits_single])
            
            x_mix = self.aug(self.mel(x_mix))
            logits_mix = self.model(x_mix)

            LS = self.loss_fn(logits_single, y, reduction='mean')
            LM = self.loss_fn(logits_mix, y_mix, reduction='mean')
            LA = self.loss_fn(logits_mix, logits_single_mix.sigmoid().detach(), reduction='mean')
            L = LS + LM + (0.5 + lam) * LA
            L.backward()
            
            self.optimizer.step()

            Loss += L.detach().item()
            

        return Loss
            
                
    def validate(self):
        self.model.eval()

        n_steps = len(self.val_loader)

        if self.config['verbose']==2: pbar = tqdm(enumerate(self.val_loader), total=n_steps, desc="Validation")
        else: pbar = enumerate(self.val_loader)

        pred = []
        target = []
        soundscapes = []
        losses = []

        with torch.no_grad():
            for batch_idx, (x, y) in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

                x = self.mel(x)

                logits = self.model(x)
                loss = self.loss_fn(logits, y, reduction='none').mean(dim=-1)

                losses.append(loss.detach().cpu().numpy())
                pred.append(logits.sigmoid().detach().cpu().numpy())
                target.append(y.detach().cpu().numpy())

        target = np.concat(target)
        pred = np.concat(pred)
        losses = np.concat(losses)
        scores = []
        
        for m in self.config['metrics']:
            scores.append(m(target, pred))

        
        return scores, losses, target, pred, soundscapes

    
    
    def train(self, epochs=16, checkpoint_freq='once', Lab=None, soup=3):
        set_seed(self.config['seed'])

        self.train_ds = XCDataset(is_train=True, fold=self.fold, config=self.config)
        self.val_ds = XCDataset(is_train=False, fold=self.fold, config=self.config)
        
        self.train_loader = self.train_ds.get_loader()
        self.val_loader = self.val_ds.get_loader()
        self.config['num_labels'] = len(self.train_ds.LABELS)
        self.epochs = epochs

        self.model = self.config['model'](config=self.config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs, eta_min=1e-8)
        self.loss_fn = self.config['loss']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.past_weights = list()
        
        best = (np.inf,0,0)
            
        Epochs = range(epochs)
        for epoch in Epochs:
            torch.manual_seed(self.config['seed']+epoch)
            
            train_loss = self.train_one_epoch(epoch=epoch)
            self.past_weights.append(self.model.state_dict())
            if soup and len(self.past_weights)>soup: self.past_weights.pop(0)
            
            if not self.config['train_only']: 
                val_scores, val_loss, target, pred, soundscapes = self.validate()
                val_loss = np.mean(val_loss)
                if val_loss<best[0]: best = (val_loss, val_scores,epoch)
            else: val_scores, val_loss = [None] * len(self.config['metrics']), None


            self.history.loc[len(self.history)] = [self.exp_id, epoch, train_loss, val_loss, self.optimizer.param_groups[0]['lr'], str(datetime.datetime.now()), self.fold, *val_scores]+list(self.config.values())
            self.history.to_csv(f'history/{self.exp_id}.csv', index=False)
            if checkpoint_freq=='epoch': torch.save(self.model.state_dict(), f"models/{self.exp_id}_{epoch}.pth")

            clear_output(wait=False)
            print(self.exp_id, '\n')
            print(f"\033[1m Epoch {epoch+1}/{epochs}")
            print(f'\033[1m Training \t|\t loss={np.round(train_loss, 3)}' + '\033[0m')
            if not self.config['train_only']:
                print(f'\033[1m Validation \t|\t loss={np.round(val_loss, 3)}  -  ' + '  -  '.join([f'{m.__name__}={np.round(s,3)}' for m,s in zip(self.config['metrics'], val_scores)])+'\033[0m')
                print()
                print(f"\033[1m Best : {'  -  '.join([f'{m.__name__}={np.round(s,3)}' for m,s in zip(self.config['metrics'], best[1])])} at epoch {best[2]}")

                

            if self.config['scheduler']: self.scheduler.step()

    
        if checkpoint_freq=='once': torch.save(self.model.state_dict(), f"models/{self.exp_id}.pth")

        if soup:
            S = {}
            for k in self.model.state_dict().keys():
                S[k] = sum([m[k] for m in self.past_weights])/len(self.past_weights)
            torch.save(S, f"models/{self.exp_id}.pth")
            
    
        if Lab: Lab.add_model([self.exp_id, epochs, train_loss, val_loss, str(datetime.datetime.now()), self.fold, *val_scores]+[repr(x) for x in self.config.values()], self.config)

        return self.model