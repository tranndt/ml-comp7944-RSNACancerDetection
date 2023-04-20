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
        self.rnn_specific_cols = ["pred"]

    def __getitem__(self, index):
        MAX_IMGS = 9
        if torch.is_tensor(index):
            index = index.tolist()
        group_key = self.group_keys[index]
        group = self.df.get_group(group_key)
        is_cancer = group['cancer'].mean()            
        row = group.iloc[0]
        row = row.copy()
        if self.ret_type == 'avg':
            row['pred'] = group['pred'].mean()
        elif self.ret_type == 'amm':
            row['pred'] = group['pred'].mean()
            row['pred_min'] = group['pred'].min()
            row['pred_max'] = group['pred'].max()
        elif self.ret_type == 'amms':
            row['pred'] = group['pred'].mean()
            row['pred_min'] = group['pred'].min()
            row['pred_max'] = group['pred'].max()
            row['pred_std'] = group['pred'].std()
            if np.isnan(row['pred_std']):
                row['pred_std'] = 0
        elif self.ret_type == 'pad':
            for i in range(MAX_IMGS):
                row['pred_' + str(i)] = 0
            i=0
            for idx, g_row in group.iterrows():          
                row['pred_' + str(i)] =  g_row['pred']
                i+=1
        elif self.ret_type == 'rnn':
            imgs = torch.tensor([]) # row[i] for i in self.rnn_specific_cols])
            count = 0
            for idx, row in group.iterrows():     
                row_info = [row[i] for i in self.rnn_specific_cols]      
                imgs = torch.cat([imgs, torch.tensor(row_info, dtype=torch.float32).unsqueeze(0)], dim=0)
                count += 1
            row = row.drop(self.exclude_cols).astype(float)
            row = row.drop(self.rnn_specific_cols).astype(float)
            row = torch.tensor(row.values, dtype=torch.float32)
            pad = torch.zeros((MAX_IMGS - imgs.shape[0], imgs.shape[1]))
            imgs = torch.cat([imgs, pad])
            return row, imgs, torch.tensor(count, dtype=torch.long), int(is_cancer)
        row = row.drop(self.exclude_cols).astype(float)
        return torch.tensor(row.values, dtype=torch.float32), int(is_cancer)

    def __len__(self):
        return self.data_len