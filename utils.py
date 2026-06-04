import os
import shutil
import numpy as np
import pandas as pd
import time
import hashlib
import torch
import random
import math

from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score
from torch.optim.lr_scheduler import LambdaLR


os.makedirs('deliveries', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('history', exist_ok=True)
os.makedirs('labs', exist_ok=True)
os.makedirs('error_analyses', exist_ok=True)

PATH = '../../Datasets/BirdCLEF/2026/'
taxonomy = pd.read_csv(PATH+'taxonomy.csv')
taxonomy.index = taxonomy.primary_label.values



def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def error_analysis(y_true, y_pred, LABELS=[]):
    y = (y_true>0).astype(int)
    p = np.round(y_pred).astype(int)
    if len(LABELS)==0:
        LABELS = np.unique(taxonomy.primary_label.values)

    DF = []
    for i, (true, pred) in enumerate(zip(y.T, p.T)):
        if np.sum(true)>0:
            precision = precision_score(true, pred) if np.sum(pred)>0 else 0.
            recall = recall_score(true, pred) if np.sum(pred)>0 else 0.
            color = ''
            if (precision+recall)/2 < .6: color = '\033[91m'
            else: color = '\033[92m'
            if precision < .6: pcolor = '\033[91m'
            else: pcolor = '\033[92m'
            if precision < .6: pcolor = '\033[91m'
            else: pcolor = '\033[92m'
            if recall < .6: rcolor = '\033[91m'
            else: rcolor = '\033[92m'
            neutral = '\033[0m'

            spaces = ''.join([' ']*int(5-np.log10(np.sum(true))))
            
            print(f'{pcolor} {np.round(precision, 3)} \t {rcolor}{np.round(recall, 3)}{neutral} \t -  {np.sum(true)}{spaces} \t {np.sum(pred)} \t - {color} {taxonomy.sort_index().loc[LABELS[i], 'common_name']}, {taxonomy.sort_index().loc[LABELS[i], 'class_name']}{neutral}, {LABELS[i]}{neutral}')

            DF.append([LABELS[i], taxonomy.sort_index().loc[LABELS[i], 'class_name'], precision, recall,  np.sum(true), np.sum(pred)])

    cols = ['primary_label', 'class_name', 'precision', 'recall', 'n_true', 'n_pred']
    return pd.DataFrame(np.stack(DF), columns=cols)


def decode_config(cfg):
    for X in cfg:
        try: cfg[X] = eval(cfg[X])
        except:None
    return cfg

    
def onnx_compile(delivery_id, Model, input_size=(23,1,224,312)):
    os.makedirs(f'deliveries/{delivery_id}/onnx', exist_ok=True)
    
    history = pd.read_csv(f'deliveries/{delivery_id}/history.csv')
    
    for i, ID in enumerate(history.id):
        cfg = {x: history.loc[i, x] for x in history.columns[12:] }|{'epochs':history.loc[i, 'epoch']}
        model = Model(config=cfg)
        model.load_state_dict(torch.load(f'models/{ID}.pth', weights_only=True, map_location=torch.device('cpu')))
        model.eval()
    
        model = torch.onnx.export(model, torch.zeros(input_size), dynamo=True)
        model.optimize()
        model.save(f'deliveries/{delivery_id}/onnx/{ID}.onnx')



def make_teachers(delivery_ids, version):
    os.makedirs(f'teachers/v{version}', exist_ok=True)
    os.makedirs(f'teachers/v{version}/weights', exist_ok=True)
    Lab = []
    Hist = []
    for did in delivery_ids:
        Lab.append(pd.read_csv(f'deliveries/{did}/lab.csv'))
        Hist.append(pd.read_csv(f'deliveries/{did}/history.csv'))
        for ID, fold in zip(Hist[-1].id, Hist[-1].fold):
            shutil.copyfile(f'deliveries/{did}/weights/{ID}.pth', f'teachers/v{version}/weights/f{fold}-r0.pth')


def merge_deliveries(delivery_ids, target):
    os.makedirs(f'deliveries/{target}', exist_ok=True)
    os.makedirs(f'deliveries/{target}/weights', exist_ok=True)
    Lab = []
    Hist = []
    for did in delivery_ids:
        Lab.append(pd.read_csv(f'deliveries/{did}/lab.csv'))
        Hist.append(pd.read_csv(f'deliveries/{did}/history.csv'))
        for ID in Lab[-1].id:
            shutil.copyfile(f'deliveries/{did}/weights/{ID}.pth', f'deliveries/{target}/weights/{ID}.pth')
        
        
    pd.concat(Lab).to_csv(f'deliveries/{target}/lab.csv', index=False)
    pd.concat(Hist).to_csv(f'deliveries/{target}/history.csv', index=False)


class Lab:
    def __init__(self, name=None, load=False):
        self.id = hashlib.sha256(('lab'+str(time.time())).encode()).hexdigest()
        self.name = name
        if load: self.df = pd.read_csv(f'labs/{self.id}.csv')
        self.df = pd.DataFrame([], columns=['id'])
        

    def add_model(self, X, config):
        columns = ['id', 'epochs', 'train_loss']
        if not config['train_only']: 
            columns = columns+['val_loss']+['val_'+m.__name__ for m in config['metrics']]+['val_soundscapes'+m.__name__ for m in config['metrics']]
        columns=columns+['timestamp', 'fold']+list(config.keys())
        self.df.loc[len(self.df), columns] = X
        self.df.to_csv(f'labs/{self.id}.csv', index=False)

    def ship(self, how='last', N=1, metric='val_AUC'):
        if not(self.name): self.name = self.id
        os.makedirs('deliveries/'+self.name, exist_ok=True)
        os.makedirs('deliveries/'+self.name+'/weights', exist_ok=True)
        self.df.to_csv(f'deliveries/{self.name}/lab.csv', index=False)

        history = []
        for m_id in self.df['id']:
            m_df = pd.read_csv('history/'+m_id+'.csv')
            if how=='best': m_df.sort_values(metric)
            if isinstance(N, list): history.append(m_df.iloc[N])
            else: history.append(m_df.iloc[-N:])
        history = pd.concat(history)
        history.index = np.arange(len(history))
        history.to_csv(f'deliveries/{self.name}/history.csv')
        for i in history.index:
            try:
                shutil.copyfile(f'models/{history.loc[i, 'id']}_{history.loc[i, 'epoch']}.pth', f'deliveries/{self.name}/weights/{history.loc[i, 'id']}_{history.loc[i, 'epoch']}.pth')
            except:
                shutil.copyfile(f'models/{history.loc[i, 'id']}.pth', f'deliveries/{self.name}/weights/{history.loc[i, 'id']}.pth')
        

    def cook_soup(self, how='last', N=1, metric='val_AUC'):
        if not(self.name): self.name = self.id
        os.makedirs('soups/'+self.name, exist_ok=True)
        os.makedirs('soups/'+self.name+'/weights', exist_ok=True)
        self.df.to_csv(f'soups/{self.name}/lab.csv', index=False)

        history = []
        for m_id in self.df['id']:
            h = pd.read_csv('history/'+m_id+'.csv')
            history.append(h)
            config = decode_config({x: h.iloc[0][x] for x in h.columns[7:]})
            models = []
            for e in h.epoch[-N:]:
                m = f'models/{m_id}_{e}.pth'
                models.append(torch.load(m, weights_only=True, map_location=torch.device('cpu')))

            keys = list(models[0])
            soup = {}
            
            for k in keys:
                soup[k] = sum([m[k] for m in models])/len(models)
            torch.save(soup, f"soups/{self.name}/weights/{m_id}.pth")
            
        history = pd.concat(history)
        history.index = np.arange(len(history))
        history.to_csv(f'soups/{self.name}/history.csv')