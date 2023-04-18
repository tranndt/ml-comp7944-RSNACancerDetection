import pandas as pd
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import numpy as np

# site_id,patient_id,image_id,laterality,view,age,cancer,biopsy,invasive,BIRADS,implant,density,machine_id,difficult_negative_case
# site_id,patient_id,image_id,laterality,view,age,implant,machine_id,prediction_id

class PredictionDataset(Dataset):
    # individual = True if you want to get the individual images
    # get_cancer = True if you want to get the cancer value vs difficult case value
    def __init__(self, csv_path:str, ret_type='avg'):
        tile = True # make tile always true so we can use pretrained
        self.df = pd.read_csv(csv_path)
        self.df = self.df.groupby(['patient_id', 'laterality_L'])
        self.group_keys = list(self.df.groups.keys())
        self.data_len = len(self.df)
        self.ret_type = ret_type
        self.exclude_cols = ['patient_id', 'image_id', 'cancer', 'biopsy', 'invasive', 'difficult_negative_case', 'BIRADS', 'density', 'implant']

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        group_key = self.group_keys[index]
        group = self.df.get_group(group_key)
        is_cancer = group['cancer'].mean()
        if self.ret_type == 'avg':
            pred = group['pred'].mean()
        row = group.iloc[0]
        row = row.copy()
        row['pred'] = pred
        row = row.drop(self.exclude_cols).astype(float)
        return torch.tensor(row.values, dtype=torch.float32), int(is_cancer)

    def __len__(self):
        return self.data_len