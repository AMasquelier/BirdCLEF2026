import torch
import torchaudio
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torch
import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchaudio, torchvision
from sklearn.model_selection import StratifiedKFold




DATA_VERSIONS = {
    # 'clean1':['cleaned_audio', 'train_clean.csv'],
    'clean1':['cleaned_audio_1_1', 'train_clean_1_1.csv'],
    'clean2':['cleaned_audio2', 'train_clean2.csv'],
    'processed':['processed_audio_2', 'train_processed2.csv'],
}
    

class BirdDataset(Dataset):
    PATH = '../../Datasets/BirdCLEF/2026/'
    
    config = {"padding":"random", "sampling":"energy", "sr":32000, 'seed':2, 'train_only':False, 'extra_data':False, 'soundscape_data':True, 'synonymous_taxa':True, 'sampling_gamma':0.5, 'batch_size':32, 'num_workers':0, 'data_version':'processed'}

    def get_site(self, x):
        try:
            x = x.split('_')
            site = x[-5]
        except:
            site = None
        return site
    
    def __init__(self, is_train=True, fold=0, config={}):
        self.is_train = is_train
        self.config.update(config)

        tax = pd.read_csv(self.PATH+'taxonomy.csv')

        self.SND_PATH = self.PATH + f'{DATA_VERSIONS[self.config['data_version']][0]}/'
        self.LABELS = list(np.unique(tax.primary_label.dropna()))

        df = pd.read_csv(self.PATH+DATA_VERSIONS[self.config['data_version']][1])
        df['secondary_labels'] = df['secondary_labels'].apply(eval)
        df['filename'] = self.SND_PATH+df['filename']
        df['src'] = 'original'

        no_call = pd.read_csv(self.PATH+f'no_call.csv')
        no_call['filename'] = self.PATH+'train_soundscapes/'+no_call['filename']+'.ogg'
        no_call['primary_label'] = no_call.filename.apply(lambda x: [])
        no_call['secondary_labels'] = no_call.filename.apply(lambda x: [])
        no_call['hash'] = no_call['filename'].values # to change
        no_call['src'] = 'no_call'

        syndf = pd.read_csv(self.PATH+f'synonyms_data_clean.csv')
        syndf['primary_label'] = syndf['primary_label'].apply(lambda x: [x])
        syndf = syndf[(syndf.status=='OK') & (syndf.hash>0)]
        syndf['secondary_labels'] = syndf['secondary_labels'].apply(eval)
        syndf['src'] = 'synonym'

        ssdf = pd.read_csv(self.PATH+f'train_soundscapes_clean.csv')
        ssdf['primary_label'] = ssdf['primary_label'].apply(eval)
        ssdf = ssdf[ssdf.status=='OK']
        ssdf['secondary_labels'] = '[]'
        ssdf['secondary_labels'] = ssdf['secondary_labels'].apply(eval)
        ssdf['filename'] = self.PATH+'cleaned_audio/'+ssdf['filename']
        ssdf['src'] = 'soundscapes'

        
        df = pd.concat([df, ssdf, syndf, no_call])
        df = df.drop_duplicates(subset=['hash'], keep='first').drop_duplicates(subset=['filename'], keep='first')
        df.index = df.filename.values

        self.cp = df.copy()

        
        IDX = np.unique(df[df['src']=='original'].index)
        np.random.seed(self.config['seed'])
        np.random.shuffle(IDX)

        if self.config['train_only']: idx = IDX
        else:
            skf = StratifiedKFold(n_splits=5)
            FOLDS = list(skf.split(IDX, df.loc[IDX].primary_label.astype(str).fillna('none').values))
            train_idx = IDX[FOLDS[fold][0]].tolist()
            val_idx = IDX[FOLDS[fold][1]].tolist()
            idx = train_idx if is_train else val_idx
            
        DF = df.loc[idx].copy()

        counts = DF['primary_label'].value_counts()
        class_weights = (counts / counts.sum())**(-self.config['sampling_gamma'])
        DF['weights'] = DF.primary_label.map(class_weights).values
        

        DF['primary_label'] = DF['primary_label'].apply(lambda x: [x])

        DF = pd.concat([DF, df[df.src=='no_call']])
        
        if self.is_train and self.config['extra_data']:
            DF = pd.concat([DF, df[df.src=='additional']])

        if self.config['soundscape_data']:
            tmp = df[df.src=='soundscapes']
            if not self.config['train_only']:
                tmp['site'] = tmp.filename.apply(self.get_site)
                tmp['recording_id'] = tmp.filename.apply(lambda x: '_'.join(x.split('/')[-1].split('_')[:-2]))
                tmp.index = tmp.recording_id.values
    
                if not(self.config['train_only']):
                    s22 = tmp[(tmp.site=='S22')]
                    s08 = tmp[(tmp.site=='S08')]
                    tmp = tmp[(tmp.site!='S22')&(tmp.site!='S08')]
        
                    
                    s22_idx = np.unique(s22.index.values.copy())
                    s08_idx = np.unique(s08.index.values.copy())
                    np.random.shuffle(s22_idx)
                    s22_idx = np.roll(s22_idx, fold*int(0.3*len(s22_idx)))
                    
                    if self.is_train:
                        tmp = tmp[(tmp.site!='S03') & (tmp.site!='S18')]
                        tmp = pd.concat([tmp, s22.loc[s22_idx[:int(0.7*len(s22_idx))]], s08.loc[s08_idx[1:]]])
                    else:
                        tmp = tmp[(tmp.site=='S03') | (tmp.site=='S18')]
                        tmp = pd.concat([tmp, s22.loc[s22_idx[int(0.7*len(s22_idx)):]], s08.loc[s08_idx[[0]]]])

            self.ss_df = tmp.copy()
            DF = pd.concat([DF, tmp])
            

        if self.is_train and self.config['synonymous_taxa']:
            DF = pd.concat([DF, df[df.src=='synonym']])

        DF['weights'] = DF['weights'].fillna(np.mean(class_weights))
        self.DF = DF
        

        self.sampler = torch.utils.data.WeightedRandomSampler(
            weights=DF['weights'].values,
            num_samples=len(DF['weights']),
            replacement=True
        )


        self.paths = list(DF['filename'].values)
        labels = DF['primary_label'].apply(self.make_labels).values
        secondary_labels = DF['secondary_labels'].apply(self.make_labels).values
        self.labels = (labels | secondary_labels)#.astype(int)
        self.src_weights = torch.ones(234)
        self.src_weights[tax[tax.primary_label.str.contains('47158son')].index] = 0.1

        

    def get_loader(self):
        if self.is_train:
            return DataLoader(
                self,
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                pin_memory=False,
                drop_last=True,
                sampler=self.sampler
            )
        else:
            return DataLoader(
                self,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=False,
            )

        
    def make_labels(self, X):
        out = np.zeros(len(self.LABELS)).astype(bool)
        for x in X:
            if x in self.LABELS:
                out[self.LABELS.index(x)] = True
            
        return out
        

    def load_sound(self, filepath, start=0, DUR=5*32000):
        wav, sr = torchaudio.load(filepath)
        wav = wav[0]

        l = len(wav)
        if l < DUR:
            padding = self.config['padding']
            if padding=='random': padding = np.random.choice(['cycle', 'zero'])
                
            if padding=='cycle':
                n_repeat = int(np.ceil(DUR/l))
                wav = torch.roll(wav.repeat(n_repeat), np.random.randint(l))[:DUR]
            else:
                wav2 = torch.zeros((DUR))
                s = np.random.randint(DUR-l)
                wav2[s:s+l] = wav
                wav = wav2
        else:
            if self.is_train:
                if self.config['sampling']=='energy':
                    dur = DUR
                    wav_hp = torchaudio.functional.highpass_biquad(wav, 32000, 512)
                    E = wav_hp**2
                    E = E.clamp(0, E.mean()+16*E.var())
                    step = dur//8
                    E_win = E.unfold(0, dur, step).mean(dim=-1)
                    p = (E_win/E_win.sum())**1.5
                    
                    idx = torch.multinomial(p, 1).item()
                    offset = wav.size(0)-((E_win.size(0)-1)*step+dur)
                    start = idx * step
                    if offset>0: start += np.random.randint(offset)
                    wav = wav[start : start + dur]
                else:
                    s = random.randint(0, l-DUR)
                    wav = wav[s:s+DUR]
            else:
                wav = wav[:DUR]
                
        return wav.clone()
        

    def __len__(self):
        return len(self.paths)
        

    def __getitem__(self, idx):
        
        path = self.paths[idx]
        DUR = int(5 * self.config['sr'])

        audio = self.load_sound(path, DUR=DUR)
        labels = self.labels[idx]
        weights = torch.ones(234) if 'BC2026' in path else self.src_weights
        
        if self.is_train:
            return (
                audio,
                weights.float(),
                torch.tensor(labels, dtype=torch.float32)
            )
        else:
            return (
                audio,
                'BC2026' in path,
                torch.tensor(labels, dtype=torch.float32)
            )


class BirdDatasetUnlabeled(Dataset):
    PATH = '../../Datasets/BirdCLEF/2026/'
    UNLABELED_PATH = PATH + 'train_soundscapes/'

    config = {"duration":5, "sr":32000, 'seed':2}

    def get_site(self, x):
        x = x.split('_')
        site = x[-3]
        return site

    def get_time(self, x):
        x = x.split('_')
        time = int(x[-1][:2]) + int(x[-1][2:4]) / 60
        return int(np.round(time))%24
    
    def get_date(self, x):
        x = x.split('_')
        time = int(x[-2][4:6]) + int(x[-2][6:8]) / 30.41
        return int(np.round(time))%12
        

    def __init__(self, config={}):
        self.config.update(config)
        df = pd.read_csv(self.PATH+f'train_soundscapes_clean.csv')
        df['filename'] = df['filename'].apply(lambda x: '_'.join(x.split('_')[:-2]))+'.ogg'
        self.paths = [self.UNLABELED_PATH+x for x in os.listdir(self.UNLABELED_PATH) if ('.ogg' in x and not(x in df.filename.values))]
        self.paths2 = [self.UNLABELED_PATH+x for x in os.listdir(self.UNLABELED_PATH) if '.ogg' in x]


    def load_sound(self, filepath, start=0):
        DUR = self.config['duration'] * self.config['sr']
        wav, sr = torchaudio.load(filepath)   # full load, no frame_offset
        wav = wav[0, start:start+DUR]
                
        return wav.clone()
        

    def __len__(self):
        return len(self.paths)
        

    def __getitem__(self, idx):
        path = self.paths[idx]
    
        start = random.randint(0, (60-self.config['duration']) * self.config['sr'])
        unlabeled = self.load_sound(path, start)

        site = self.get_site(path)
        time = self.get_time(path)
        date = self.get_date(path)
        return unlabeled, (site, time, date)
        

class XCDataset(Dataset):
    PATH = "../../Datasets/Xeno-Canto/"
    
    config = {"padding":"random", "sr":32000, 'seed':2, 'train_only':False}

    def __init__(self, is_train=True, fold=0, config={}):     
        self.config.update(config)

        df = pd.read_csv(self.PATH+'files_details.csv', low_memory=False)
        
        df['license'] = df.lic.apply(lambda x: x.split('/')[-3])
        df = df[df.sp!='mystery']
        df = df[df.license!='by-nc-nd']
        df = df[df.license!='3.0']
        df['filename'] = self.PATH+'audio/'+df['sp'].astype(str)+'/XC'+df['id'].astype(str)+'.ogg'
        df['status'] = df.filename.apply(lambda x: 'OK' if os.path.exists(x) else None)
        df = df.dropna(subset=['status']).drop_duplicates(subset=['filename', 'sp', 'gen'])

        df['primary_label'] = df.gen.apply(lambda x: x).values
        df['secondary_labels'] = df['also'].apply(eval)
        df['secondary_labels'] = df['secondary_labels'].apply(lambda x: [y.split(' ')[0] for y in x])

        sp_counts = df.groupby('primary_label').count()
        df.index = df.primary_label.values
        df = df.loc[sp_counts[sp_counts>32].index]
        
        self.LABELS = list(np.unique(df.primary_label))
            
        df.index = df.filename.values
        
        IDX = np.unique(df.index)
        np.random.seed(self.config['seed'])
        np.random.shuffle(IDX)

        if self.config['train_only']: idx = IDX
        else:
            skf = StratifiedKFold(n_splits=20)
            FOLDS = list(skf.split(IDX, df.loc[IDX].primary_label.fillna('none').values))
            train_idx = IDX[FOLDS[fold][0]].tolist()
            val_idx = IDX[FOLDS[fold][1]].tolist()
            idx = train_idx if is_train else val_idx
            
        DF = df.loc[idx].copy()

        self.paths = list(DF['filename'].values)#.str.replace('.ogg','.mp3').values)
        labels = DF['primary_label'].apply(self.make_labels).values
        secondary_labels = DF['secondary_labels'].apply(self.make_secondary_labels).values
        self.labels = (labels | secondary_labels)#.astype(int)
        
        self.is_train = is_train
        self.DF = DF
        

    def make_labels(self, X):
        out = np.zeros(len(self.LABELS)).astype(bool)
        out[self.LABELS.index(X)] = True
            
        return out

    def make_secondary_labels(self, X):
        out = np.zeros(len(self.LABELS)).astype(bool)
        for x in X:
            if x in self.LABELS:
                out[self.LABELS.index(x)] = True
            
        return out
        

    def load_sound(self, filepath, start=0, DUR=5*32000):
        wav, sr = torchaudio.load(filepath)
        wav = wav[0]

        l = len(wav)
        if l < DUR:
            padding = self.config['padding']
            if padding=='random': padding = np.random.choice(['cycle', 'zero'])
                
            if padding=='cycle':
                n_repeat = int(np.ceil(DUR/l))
                wav = torch.roll(wav.repeat(n_repeat), np.random.randint(l))[:DUR]
            else:
                wav2 = torch.zeros((DUR))
                s = np.random.randint(DUR-l)
                wav2[s:s+l] = wav
                wav = wav2
        else:
            if self.is_train:
                s = random.randint(0, l-DUR)
                wav = wav[s:s+DUR]
            else:
                wav = wav[:DUR]
                
        return wav

    def get_loader(self):
        if self.is_train:
            return DataLoader(
                self,
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                pin_memory=True,
                drop_last=True,
                shuffle=True,
            )
        else:
            return DataLoader(
                self,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=True,
            )

    def __len__(self):
        return len(self.paths)
        

    def __getitem__(self, idx):
        path = self.paths[idx]
        DUR = int(self.config['duration'] * self.config['sr'])

        audio = self.load_sound(path, DUR=DUR)
        labels = self.labels[idx]
        
        return (
            audio,
            torch.tensor(labels, dtype=torch.float32)
        )


