import pandas as pd
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import io



class MammogramDataset(Dataset):
    # individual = True if you want to get the individual images
    # get_cancer = True if you want to get the cancer value vs difficult case value
    def __init__(self, csv_path:str, data_path:str='processed_data/', transform=None, individual=False, get_cancer=True):
        self.transform = transform
        self.df = pd.read_csv(csv_path)
        self.data_len = len(self.df)
        self.data_path = data_path
        self.individual = individual
        self.get_cancer = get_cancer
        if not individual:
            self.data_len = self.df['patient_id'].nunique()
            self.patient_list = self.df['patient_id'].unique()

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        imgs = None
        target_name = 'cancer'
        if self.individual:
            patient_id, image_id, target = self.df.loc[index, ['patient_id', 'image_id', target_name]]
            if not self.get_cancer:
                target = max(target, int(self.df.loc[index, 'difficult_negative_case']))
            image_location = str(self.data_path) + "/" + str(int(patient_id)) + "_" + str(int(image_id)) + ".png"
            imgs = Image.open(image_location)
            if self.transform:
                imgs = self.transform(imgs)
        else:
            selected_patient = self.patient_list[index]
            relevant_rows = self.df.loc[self.df['patient_id'] == selected_patient]
            imgs = []
            for row in relevant_rows.iterrows():
                image_id = int(row.iloc['image_id'])
                target = row.iloc[target_name]
                if not self.get_cancer:
                    target = max(target, int(row.iloc[index, 'difficult_negative_case']))
                img = Image.open(self.data_path + "/" + selected_patient + "_" + image_id + ".png")
                if self.transform:
                    img = self.transform(img)
                    imgs.append(img)
        return imgs, int(target)

    def __len__(self):
        return self.data_len