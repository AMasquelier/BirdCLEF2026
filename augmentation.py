import numpy as np
import pandas as pd
import torch
import torch.nn as nn


import torchvision.transforms.functional as TVF
import torchaudio
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch

from scipy import signal



class PitchShift(nn.Module):
    def __init__(self, minimum=0, maximum=True):
        super().__init__()
        self.min = minimum
        self.max = maximum
        self.transform = transforms.PitchShift(sample_rate, 4)
        
    def forward(self, x):

        
        noise = torch.randn(x.size()).to(x.device)
        if self.rand_std:
            std = torch.rand(x.size(0)).to(x.device)*self.std
            std = std.view(-1, 1, 1, 1).float()
        else: std = self.std
        x = x + noise * std
        x = (x - x.min()) / (x.max() - x.min())
        return x

class RandomGain(nn.Module):
    def __init__(self, rng=(-6,6)):
        super().__init__()
        self.range = rng
        
    def forward(self, x):
        gain_db = self.range[0] + torch.rand(x.size(0)).unsqueeze(1) * (self.range[1]-self.range[0])
        ratio = 10. ** (gain_db.to(x.device) / 20.)
        return x * ratio

        

class GaussianNoise(nn.Module):
    def __init__(self, mean=0, std=.5, rand_std=True):
        super().__init__()
        self.mean = mean
        self.std = std
        self.rand_std = rand_std
        
    def forward(self, x):
        noise = torch.randn_like(x).to(x.device)
        std = x.max()*self.std
        if self.rand_std:
            std = torch.rand(x.size(0)).to(x.device)*std
            std = std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else: std = std

        scale = torch.arange(x.size(2),0,-1).unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(x.device) / x.size(2)
        
        B, C = noise.shape[:2]
        flat = noise.view(B, C, -1)
        mins = flat.min(dim=-1).values[..., None, None]
        maxs = flat.max(dim=-1).values[..., None, None]
        noise = (noise - mins) / (maxs - mins + 1e-7)
        gamma = torch.rand(B).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(x.device)
        
        x = x + noise  * std * scale**gamma
        
        B, C = x.shape[:2]
        flat = x.view(B, C, -1)
        mins = flat.min(dim=-1).values[..., None, None]
        maxs = flat.max(dim=-1).values[..., None, None]
        x = (x - mins) / (maxs - mins + 1e-7)
        return x
        

class Stretch(nn.Module):
    def __init__(self, factor=0.1, axis='both'):
        super().__init__()
        self.factor = factor
        self.axis = axis
        
    def forward(self, x):
        target_size = x.shape[-2:]  # (H, W) from the input
        
        for i in range(x.size(0)):
            img = x[i]
            
            if self.axis in ('both', 'y'):
                shift_factor = torch.rand(1) * self.factor
                shift = int(shift_factor * x.size(-2))
                if shift > 0: img = img[:, shift:]
                elif shift < 0: img = img[:, :shift]
            if self.axis in ('both', 'x'):
                shift_factor = torch.rand(1) * self.factor
                shift = int(shift_factor * x.size(-1))
                if shift > 0: img = img[:, :, shift:]
                elif shift < 0: img = img[:, :, :shift]
            
            x[i] = TVF.resize(img, target_size, TVF.InterpolationMode.BILINEAR, None, True)
            
        return x


class PinkNoise(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, noise):
        batch_size = x.size(0)
        lam = torch.tensor(np.random.beta(1.2,1.2, batch_size)).to(x.device) * .5
        gate = torch.rand(batch_size).to(x.device).round()
        Lam = (lam * gate).unsqueeze(-1)
        x = x * (1-Lam) + noise * Lam 
        return x


        
class Shift(nn.Module):
    def __init__(self, factor=0, axis='both'):
        super().__init__()
        self.factor = factor
        self.axis = axis
        
    def forward(self, x):
        x = x.clone()
        
        for i in range(x.size(0)):
            img = x[i]
            
            if self.axis=='both' or self.axis=='y':
                shift_factor = (torch.rand(1)*2-1)*self.factor
                size = x.size(-2)
                shift = int(shift_factor*size)
                if shift>0: 
                    img[:,:size-shift] = img[:,shift:].clone()
                    img[:,size-shift:] *= 0
                elif shift<0: 
                    img[:,-shift:] = img[:,:shift].clone()
                    img[:,:-shift] *= 0

            if self.axis=='both' or self.axis=='y':
                shift_factor = (torch.rand(1)*2-1)*self.factor
                size = x.size(-1)
                shift = int(shift_factor*size)
                if shift>0: 
                    img[:,:,:size-shift] = img[:,:,shift:].clone()
                    img[:,:,size-shift:] *= 0
                elif shift<0: 
                    img[:,:,-shift:] = img[:,:,:shift].clone()
                    img[:,:,:-shift] *= 0

            img = (img-img.mean())/(torch.std(img)+1e-7)
            img = (img-img.min())/(img.max()-img.min)
            
            x[i] = img
        return x
        

class KMixAudio(nn.Module):
    def __init__(self, alpha=0.5, theta=1):
        super().__init__()
        self.alpha = alpha
        self.theta = theta
        self.N = 8

    def forward(self, x, y, tau=1):
        batch_size = x.size(0)

        lam = torch.tensor(np.random.dirichlet([2]*self.N, batch_size)).to(x.device)
        gate = torch.ones_like(lam)
        gate[:,2:] = gate[:,2:] * (torch.rand_like(gate[:,2:])>tau)
        gate = gate.sort(dim=-1, descending=True)[0].float()

        lam = lam * gate
        lam = lam / lam.sum(dim=-1, keepdims=True)
        lam = lam.sort(dim=-1, descending=True)[0].float()

        idx = torch.stack([torch.randperm(batch_size) for _ in range(self.N)]).T.to(x.device)
        idx[:,0] = torch.arange(batch_size)
        idx = (idx * gate + torch.arange(batch_size).unsqueeze(-1).to(x.device) * (1-gate)).int()

        x = (x[idx] * lam.unsqueeze(dim=-1)).sum(dim=1) / lam.pow(2).sum(dim=1).sqrt().unsqueeze(dim=-1)

        if isinstance(y, list):
            for i in range(len(y)):
                y[i] = y[i][idx].max(dim=1)[0]
        else:
            y = y[idx].max(dim=1)[0]

        return x, y

class MixupAudio(nn.Module):
    def __init__(self, alpha=0.5, theta=1, mode='max'):
        super().__init__()
        self.alpha = alpha
        self.theta = theta
        self.mode = mode

    def forward(self, x, y, n=None):
        batch_size = x.size(0)

        if n is None: n = np.random.randint(1,4)
    
        idx = torch.randperm(batch_size).to(x.device)
        
        lam = torch.tensor(np.random.beta(self.alpha,self.alpha, batch_size)).to(x.device)
        lam = torch.maximum(lam, 1-lam).float().unsqueeze(-1)
        
        x = (lam * x + (1-lam) * x[idx])# / torch.sqrt(lam**2+(1-lam)**2)
        if isinstance(y, list):
            for i in range(len(y)):
                if self.mode=='max':
                    y[i] = torch.maximum(y[i], y[i][idx])
                else:
                    y[i] = (lam * y[i] + (1-lam) * y[i][idx])
        else:
            if self.mode=='max': 
                y = torch.maximum(y, y[idx])
            else:
                y = (lam * y + (1-lam) * y[idx])
        
        return x, y


class Mixup(nn.Module):
    def __init__(self, alpha=0.5, theta=1):
        super().__init__()
        self.alpha = alpha
        self.theta = theta

    def forward(self, x, y):
        batch_size = x.size(0)
        
        lam = torch.tensor(np.random.beta(self.alpha,self.alpha, batch_size)).to(x.device)
        lam = torch.maximum(lam, 1-lam).float()
        
        idx = torch.randperm(batch_size).to(x.device)

        x = lam[...,None, None, None]* x + (1 - lam)[...,None, None, None] * x[idx]
        if isinstance(y, list): 
            for i in range(len(y)):
                y[i] = lam[...,None] * y[i] + (1 - lam)[...,None] * y[i][idx]
                y[i][y[i]>=self.theta] = 1
        else: 
            y = lam[...,None] * y + (1 - lam)[...,None] * y[idx]
            y[y>=self.theta] = 1
        
        return x, y


        

class Masking(nn.Module):
    def __init__(self, shape='square', N=4, min_freq_band=1, max_freq_band=32, min_time_band=1, max_time_band=32, mask_value=0, N_dist='fixed'):
        super().__init__()
        self.shape = shape
        self.N = N
        self.min_freq_band = min_freq_band
        self.max_freq_band_h = max_freq_band-min_freq_band
        self.min_time_band = min_time_band
        self.max_time_band_h = max_time_band-min_time_band
        self.mask_value = mask_value
        self.N_dist = N_dist

    def forward(self, x):
        return self.masking(x)
        
    def masking(self, x):
        batch_size = x.size(0)
        h = x.size(2)
        w = x.size(3)
        channels = x.size(1)
        device = x.device
        x = x.clone()

        if self.N_dist=='random':
            N = np.random.randint(self.N)
        else:
            N = self.N
        
        W = torch.tensor(self.min_time_band+np.random.beta(1,1,(N,batch_size))*self.max_time_band_h).to(device=device, dtype=torch.int32).unsqueeze(-1)
        H = torch.tensor(self.min_freq_band+np.random.beta(1,1,(N,batch_size))*self.max_freq_band_h).to(device=device, dtype=torch.int32).unsqueeze(-1)
        X = torch.tensor(np.random.beta(1,1,(N,batch_size))*w).to(device=device, dtype=torch.int32).unsqueeze(-1)
        Y = torch.tensor(np.random.beta(1,1,(N,batch_size))*h).to(device=device, dtype=torch.int32).unsqueeze(-1)

        batch_indices_x = torch.arange(w, device=x.device).unsqueeze(0)
        batch_indices_y = torch.arange(h, device=x.device).unsqueeze(0)
        
        mask_x = (batch_indices_x >= X) & (batch_indices_x < X + W)
        mask_y = (batch_indices_y >= Y) & (batch_indices_y < Y + H)
        
        mask_x = mask_x.unsqueeze(2).unsqueeze(3).repeat(1, 1, channels, h, 1)
        mask_y = mask_y.unsqueeze(2).unsqueeze(4).repeat(1, 1, channels, 1, w)

        if self.shape=='square':
            mask = ((mask_x & mask_y).sum(dim=0)>0)
        elif self.shape=='cross':
            mask = ((mask_x | mask_y).sum(dim=0)>0)
        elif self.shape=='horizontal': mask = (mask_y.sum(dim=0)>0)
        elif self.shape=='vertical': mask = (mask_x.sum(dim=0)>0)
        else: mask = (mask_y.sum(dim=0)>np.inf)

        if self.mask_value=='random':
            x[mask] = torch.rand([1]).to(device)
        elif self.mask_value=='gaussian':
            noise = torch.rand_like(x)
            x[mask] = noise[mask]
        else:
            x[mask] = self.mask_value
    
        return x
