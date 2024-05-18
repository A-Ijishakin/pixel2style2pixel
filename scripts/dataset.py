import torch 
import torchvision.transforms as tfs 
import numpy as np
from glob import glob
from PIL import Image 
import os 
from tqdm import tqdm 
import pandas as pd      

def pre_process(path, size=64): 
    transform = tfs.Compose([
            tfs.ToTensor(), 
            tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ])
    
    x = Image.open(path) 

    # resize the image
    x = x.resize((size, size))    
    
    x = transform(x)

    return x 

class CelebA_Dataset(torch.utils.data.Dataset):
    def __init__(self, mode=0, dataset='celeba'):
        #filter for those in the training set
        self.datums = pd.read_csv('/home/rmapaij/sae_bench/beta-tcvae/celeba.csv')
        self.datums = self.datums[self.datums['set'] == mode]  
        #instantiate the base directory 
        self.base = '/home/rmapaij/sae_bench/img_align_celeba' 
        
        self.size = 512 
        
    def __len__(self): 
        return len(self.datums) - 1 

    
    def __getitem__(self, idx):
        path = '{}/{}'.format(self.base, 
                self.datums.iloc[idx]['id']) 
        
        x = pre_process(path, self.size) 
                    
        labels = torch.tensor(self.datums.iloc[idx].drop(['id', 'set']).values.astype(float))
        return {'imgs': x.to(torch.float32),  
                'index' : idx, 'path': path, 'labels': labels}   
            
    
class FFHQ_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.base = '/home/rmapaij/HSpace-SAEs/datasets/FFHQ/images/*' 
        self.images = glob(self.base) 
        
    def __len__(self): 
        return len(self.images) - 1 
    
    def __getitem__(self, idx):
        x = pre_process(self.images[idx]) 
        return {'img': x, 'index' : idx, 'path': self.images[idx]}    