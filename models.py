import timm
import torch
import torch.nn as nn

class BirdModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = {
            'scale':1,
            'backbone_pooling':'avg',
            'backbone':'tf_efficientnetv2_b0',
            'dropout':0.1,
            'pretrained':True,
            'channels':1,
            'num_labels':234,
        }
        if config: self.config.update(config)

        self.training = True

        self.backbone = timm.create_model(
            self.config['backbone'], 
            pretrained=self.config['pretrained']=='imagenet',  
            num_classes=0,  
            global_pool=self.config['backbone_pooling'],
            in_chans=self.config['channels'],
            drop_rate=self.config['dropout'],
            drop_path_rate=self.config['dropout'],
        )
        feature_dim = self.backbone.num_features
        
        if self.config['pretrained']=='xc':
            self.backbone.load_state_dict(torch.load(f'pretrained_models/XC_16_{self.config['backbone']}.pth', weights_only=True))

        self.head = nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(feature_dim, 512*self.config['scale']),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(self.config['dropout']),
            torch.nn.Linear(512*self.config['scale'], self.config['num_labels'])
        )
        
        
    def forward(self, x):
        x = self.backbone(x)
        labels = self.head(x)
        return labels


