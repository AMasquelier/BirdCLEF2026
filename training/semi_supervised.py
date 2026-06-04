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

from models import BirdModel
from dataset import BirdDataset, BirdDatasetUnlabeled
from losses import CASL, ASL, BCE
from metrics import AUC, MAP, CMAP
from spectrogram import Spectrogram
from utils import Lab, set_seed, error_analysis
from augmentation import Masking, Stretch, MixupAudio


def create_dataloaders(config={'batch_size':32, 'num_workers':8, 'seed':2}, fold=0):
    train_dataset = BirdDataset(is_train=True, config=config, fold=fold)
    val_dataset = BirdDataset(is_train=False, config=config, fold=fold)
    ul_dataset = BirdDatasetUnlabeled()

    torch.manual_seed(config['seed'])
    train_loader = train_dataset.get_loader()    
    val_loader = val_dataset.get_loader()

    ul_loader = DataLoader(
        ul_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        drop_last=True,
        shuffle=True,
    )

    targets = {"labels":train_dataset.LABELS}
    
    return train_loader, val_loader, ul_loader, targets



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
    'metrics':[AUC,CMAP,MAP],
    'scheduler':True,
    'model':BirdModel,
    'num_labels':234,
    'augmentation':'light',
    'ema':0,
    'T':0.5,
}


class Trainer:
    
    def __init__(self, config={}, fold=0, epochs=16):
        self.config = CFG.copy()
        self.config.update(config)
        set_seed(self.config['seed'])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.exp_id = hashlib.sha256(str(time.time()).encode()).hexdigest()

        cols = ['id', 'epoch', 'train_loss', 'val_loss', 'lr', 'timestamp', 'fold']
        if not self.config['train_only']: cols = cols+['val_'+m.__name__ for m in self.config['metrics']]+['val_soundscape_'+m.__name__ for m in self.config['metrics']]
        cols = cols+list(self.config.keys())
        self.history = pd.DataFrame([], columns=cols)

        self.batch_size = self.config['batch_size']
        self.fold = fold

        self.aug = {
            'light': torchvision.transforms.Compose([
                Stretch(.1, axis='x'),
                Masking(N=8, max_freq_band=2, max_time_band=3, shape='cross'),
                Masking(N=8, max_freq_band=16, max_time_band=16, shape='square'),
            ]),
            'strong': torchvision.transforms.Compose([
                Stretch(.1, axis='both'),
                Masking(N=16, max_freq_band=2, max_time_band=3, shape='cross'),
                Masking(N=16, max_freq_band=16, max_time_band=16, shape='square'),
            ]),
            'extreme': torchvision.transforms.Compose([
                Stretch(.2, axis='both'),
                Masking(N=24, max_freq_band=2, max_time_band=3, shape='cross'),
                Masking(N=24, max_freq_band=16, max_time_band=16, shape='square'),
            ]),
        }

        self.mel = Spectrogram(**config['mel']).to(self.device)
        

    def train_one_epoch(self, epoch=0):
        self.model.train()        
        Loss = 0
        n_steps = len(self.train_loader)
        batch_size = self.config['batch_size']

        lam = epoch/self.epochs
        gamma = lam*2
        tau = 0.9
        
        if self.config['verbose']==2: pbar = tqdm(enumerate(self.train_loader), total=n_steps, desc="Training")
        else: pbar = enumerate(self.train_loader)

        mix_audio = MixupAudio(**self.config['mix'])
        # mix_audio = KMixAudio(**self.config['mix'])

        ul_iter = iter(self.ul_loader)

        #self.model.freeze(['backbone'])

        printed = False
        for batch_idx, (x_l, w_l, y_l) in pbar:
            self.optimizer.zero_grad()

            try:
                x_u, (site, time, month) = next(ul_iter)
            except StopIteration:
                ul_iter = iter(self.ul_loader)
                x_u, (site, time, month) = next(ul_iter)

            x_u = x_u.to(self.device)
            x_l = x_l.to(self.device)
            y_l = y_l.to(self.device) # Hard labels
            w_l = w_l.to(self.device)

            x = torch.concat([x_l, x_u])
            x_single = self.aug[self.config['augmentation']](self.mel(x))
            logits_single = self.model(x_single)
            
            with torch.no_grad():                        # Compute pseudo-labels
                ypl = torch.zeros_like(y_l)
                for teacher in self.teachers:
                    logits_teacher = teacher(self.mel(x_u))
                    T = self.config['T']
                    ypl += (logits_teacher / T).sigmoid().detach()
                ypl = ypl / len(self.teachers)
                

            w = torch.concat([w_l, torch.ones_like(w_l)])
            y = torch.concat([y_l, ypl])    # Concatenate original labels of labeled data and the pseudo-labels of the unlabeled data

            x_mix, (y_mix,w_mix,logits_single_mix) = mix_audio(x, [y,w,logits_single])

            x_mix = self.aug[self.config['augmentation']](self.mel(x_mix))
            logits_mix = self.model(x_mix)


            #LP = (self.loss_fn(logits_single, ypl, w, reduction='none')).mean()
            LS = (self.loss_fn(logits_single, y, w, reduction='none')).mean()
            LM = (self.loss_fn(logits_mix, y_mix, w_mix, reduction='none')).mean()
            LA = self.loss_fn(logits_mix, logits_single_mix.sigmoid().detach(), reduction='mean')
            L = LS + LM + LA #+ LP *.1 #* lam**0.2
            L.backward()
            
            self.optimizer.step()

            Loss += L.detach().item()
            

        return Loss
            
                
    def validate(self):
        self.model.eval()        
        Loss = 0

        n_steps = len(self.val_loader)

        if self.config['verbose']==2: pbar = tqdm(enumerate(self.val_loader), total=n_steps, desc="Validation")
        else: pbar = enumerate(self.val_loader)

        pred = []
        target = []
        soundscapes = []

        with torch.no_grad():
            for batch_idx, (x, is_ss, y) in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

                x = self.mel(x)

                logits = self.model(x)
                loss = self.loss_fn(logits, y)

                Loss += loss.detach().item()
                pred.append(logits.sigmoid().detach().cpu().numpy())
                target.append(y.detach().cpu().numpy())
                soundscapes.append(is_ss)

        target = np.concat(target)
        pred = np.concat(pred)
        soundscapes = np.concat(soundscapes)
        scores = []
        
        for m in self.config['metrics']:
            scores.append(m(target, pred))

        for m in self.config['metrics']:
            scores.append(m(target[soundscapes], pred[soundscapes]))

        
        return scores, Loss, target, pred, soundscapes

    
    def train(self, epochs=16, checkpoint_freq='once', Lab=None, soup=3):
        set_seed(self.config['seed'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_loader, self.val_loader, self.ul_loader, self.targets = create_dataloaders(fold=self.fold, config=self.config)
        self.config['num_labels'] = len(self.targets['labels'])
        self.epochs = epochs

        self.teachers = []
        for f in range(5):
            try:
                teacher = self.config['model'](config=self.config|{"backbone":self.config['teacher_backbone']})
                teacher.load_state_dict(torch.load(f'teachers/v{self.config['teacher_version']}/weights/f{f}-r{self.config['round']-1}.pth', weights_only=True))
                teacher.eval()
                teacher.to(self.device)
                self.teachers.append(teacher)
            except: self.teachers.append(None) 
        
        self.model = self.config['model'](config=self.config)
        if self.config['student_init']=='teacher':
            self.model.load_state_dict(torch.load(f'teachers/v{self.config['teacher_version']}/weights/f{self.fold}-r0.pth', weights_only=True))
            # self.model.load_state_dict(torch.load(f'teachers/v{self.config['teacher_version']}/weights/f{self.fold}-r{self.config['round']-1}.pth', weights_only=True))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs, eta_min=1e-6)
        self.loss_fn = self.config['loss']
        
        
        self.model = self.model.to(self.device)

        self.past_weights = list()
        
        best = (np.inf,0,0)
        
            
        Epochs = range(epochs)
        set_seed(self.config['seed']+self.fold*128+(self.config['round']-1)*1024)
        for epoch in Epochs:
            #torch.manual_seed(self.config['seed']+epoch)
            
            train_loss = self.train_one_epoch(epoch=epoch)
            self.past_weights.append(self.model.state_dict())
            if soup and len(self.past_weights)>soup: self.past_weights.pop(0)
            if self.config['ema']:
                for teacher in self.teachers:
                    S = {k: (1-self.config['ema']) * teacher.state_dict()[k] + self.config['ema'] * self.model.state_dict()[k] for k in teacher.state_dict().keys()}
                    teacher.load_state_dict(S)
            
            if not self.config['train_only']: 
                val_scores, val_loss, target, pred, soundscapes = self.validate()
                if val_loss<best[0]: best = (val_loss, val_scores,epoch)
            else: val_scores, val_loss = [], None

            print(self.history.shape, len([self.exp_id, epoch, train_loss, val_loss, self.optimizer.param_groups[0]['lr'], str(datetime.datetime.now()), self.fold, *val_scores]+list(self.config.values())))
            self.history.loc[len(self.history)] = [self.exp_id, epoch, train_loss, val_loss, self.optimizer.param_groups[0]['lr'], str(datetime.datetime.now()), self.fold, *val_scores]+list(self.config.values())
            self.history.to_csv(f'history/{self.exp_id}.csv', index=False)
            if checkpoint_freq=='epoch': torch.save(self.model.state_dict(), f"models/{self.exp_id}_{epoch}.pth")

            clear_output(wait=False)
            print(self.exp_id, '\n')
            print(f"\033[1m Epoch {epoch+1}/{epochs}")
            print(f'\033[1m Training \t|\t loss={np.round(train_loss, 3)}' + '\033[0m')
            if not self.config['train_only']:
                print(f'\033[1m Validation \t|\t loss={np.round(val_loss, 3)}  -  ' + '  -  '.join([f'{m.__name__}={np.round(s,3)}' for m,s in zip(self.config['metrics'], val_scores)])+'\033[0m')
                print(f'\033[1m Val soundscapes|\t' + '  -  '.join([f'{m.__name__}={np.round(s,3)}' for m,s in zip(self.config['metrics'], val_scores[3:])])+'\033[0m')
                print()
                print(f"\033[1m Best : {'  -  '.join([f'{m.__name__}={np.round(s,3)}' for m,s in zip(self.config['metrics'], best[1])])} at epoch {best[2]}")

                print(f"\nSoundscape errror analysis")
                error_analysis(target[soundscapes], pred[soundscapes])
                print(f"\nOverall errror analysis")
                error_analysis(target, pred)

            if self.config['scheduler']: self.scheduler.step()

        if not self.config['train_only']: 
            _ = self.validate()
            print(f"\nSoundscape errror analysis")
            ea_ss = error_analysis(target[soundscapes], pred[soundscapes])
            print(f"\nOverall errror analysis")
            ea = error_analysis(target, pred)
            ea_ss.to_csv(f'error_analyses/{self.exp_id}_soundscape.csv', index=False)
            ea.to_csv(f'error_analyses/{self.exp_id}.csv', index=False)
    
        if checkpoint_freq=='once': torch.save(self.model.state_dict(), f"models/{self.exp_id}.pth")

        torch.save(self.model.state_dict(), f'teachers/v{self.config['teacher_version']}/weights/f{self.fold}-r{self.config['round']}.pth')

        if soup:
            S = {}
            for k in self.model.state_dict().keys():
                S[k] = sum([m[k] for m in self.past_weights])/len(self.past_weights)
            torch.save(S, f"models/{self.exp_id}.pth")
            torch.save(S, f'teachers/v{self.config['teacher_version']}/weights/f{self.fold}-r{self.config['round']}.pth')
    
        if Lab: 
            if self.config['train_only']:
                Lab.add_model([self.exp_id, epochs, train_loss, str(datetime.datetime.now()), self.fold]+[repr(x) for x in self.config.values()], self.config)
            else:
                Lab.add_model([self.exp_id, epochs, train_loss, val_loss, str(datetime.datetime.now()), self.fold, *val_scores]+[repr(x) for x in self.config.values()], self.config)

        return self.model