import pandas as pd
from sklearn.utils import resample
import torch
import torch.nn.functional as F
import albumentations as A
import albumentations.pytorch as AP
from PIL import Image
import numpy as np
import os
from training_functions import get_model 
from patch_producer import PatchProducer
from CustomVIT import vit_b_16
from tqdm import tqdm


split_path = 'data_splits/standard/'
MODEL = "vit"
TECHNIQUE = 4

uses_patch = [False, False, True, True, True]

best_models = {
    1: {
        "vit": "cosine_True_0.0001_0.9mammograms_vit_0.621_20.pth",
        "resnet50": "cosine_True_0.001_0.9mammograms_resnet50_0.651_1.pth"
    },
    2: {
        "vit": "",
        "resnet50": ""
    },
    3: {
        "vit": "cosine_True_0.03_0.03_0.9mammograms_vit_0.653_0.pth",
        "vit_patch": "cosine_True_0.03_0.03_0.9mammograms_vitpatch_0.653_0.pth",
        "resnet50": "cosine_True_0.03_0.03_0.9mammograms_resnet50_0.675_4.pth",
        "resnet50_patch": "cosine_True_0.03_0.03_0.9mammograms_resnet50patch_0.675_4.pth"
    },
    4: {
        "vit": "cosine_True_0.0005_0.0001_0.9mammograms_vit_0.643_2.pth",
        "vit_patch": "cosine_True_0.0005_0.0001_0.9mammograms_vitpatch_0.643_2.pth",
    },
    5: {
        "vit": "",
        "resnet50": ""
    },
}


file = open(os.path.abspath(split_path+'mean_std.txt'), 'r')
mean = float(file.readline())
std = float(file.readline())
file.close()

transform_test = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0, always_apply=True),
    AP.ToTensorV2()
])


def get_pred_model():
    dir_prefix = "technique_" + str(TECHNIQUE) + "/trained_models/" + MODEL + "/"
    use_patch = uses_patch[TECHNIQUE-1]
    if TECHNIQUE == 4:
        net = vit_b_16(intermediate_embedding_size=768)
    else:
        net = get_model(MODEL)
    patch = PatchProducer() 
    net_path = best_models[TECHNIQUE][MODEL]
    if use_patch:
        patch_path = best_models[TECHNIQUE][MODEL+"_patch"]
        patch.load_state_dict(torch.load(dir_prefix + patch_path))
        patch.eval()
    net.load_state_dict(torch.load(dir_prefix + net_path))
    net.eval() 
    return net, patch    


def get_img(path):
    # load image to pytorch tensor 
    imgs = Image.open("processed_data/" + path)
    imgs =  transform_test(image=np.array(imgs))
    imgs = imgs['image']
    imgs = torch.cat((imgs, imgs, imgs), dim=0)
    imgs = imgs.expand(3, -1, -1)
    imgs = imgs.unsqueeze(0)
    return imgs


def get_pred(net, patch_net, image, meta):
    if TECHNIQUE == 1:
        outputs = net(image)
    elif TECHNIQUE == 3:
        patch = patch_net(meta)
        image[:, :, :16, 208:] = patch
        outputs = net(image)
    elif TECHNIQUE == 4:
        patch = patch_net(meta)
        patch = patch.reshape(patch.shape[0], -1)
        outputs = net(image, patch)
    else:
        raise NotImplementedError
    outputs = F.softmax(outputs, dim=1)
    outputs = outputs[:, 1].cpu().detach().numpy()
    return outputs


def get_meta(row):
    exclude_cols = ['patient_id', 'image_id', 'cancer', 'biopsy', 'invasive', 'difficult_negative_case', 'BIRADS', 'density', 'implant']
    X = row.drop(exclude_cols).astype(float)
    X = torch.tensor(X.values, dtype=torch.float32)
    return X.unsqueeze(0)


def process_preds(results_df, rows, preds):
    for i in range(len(preds)):
        rows[i]['pred'] = preds[i]
        results_df = pd.concat([results_df, rows[i]], ignore_index=True)
    return results_df


def generate_data(name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv(split_path + name)
    grouped = df.groupby(['patient_id', 'laterality_L'])
    net, patch = get_pred_model()
    net = net.to(device)
    patch = patch.to(device)
    # increase pandas print column size:
    
    # drop_cols = ['view_AT', 'view_CC', 'view_LM', 'view_ML', 'view_MLO']
    batch_size = 16
    cnt = 0
    rows = []
    imgs = torch.tensor([])
    metas = torch.tensor([])
    imgs = imgs.to(device)
    metas = metas.to(device)

    result_csv = pd.DataFrame()
    for _, group in tqdm(grouped):
        for idx, row in group.iterrows():
            imgs = torch.cat([imgs, get_img(str(row['patient_id']) + "_" + str(row['image_id']) + ".png").to(device)], dim=0)
            metas = torch.cat([metas, get_meta(row).to(device)], dim=0)
            rows.append(row.to_frame().T)
            if cnt % batch_size == 0:
                preds = get_pred(net, patch, imgs, metas)
                result_csv = process_preds(result_csv, rows, preds)
                cnt, rows, imgs, metas = 0, [], torch.tensor([]).to(device), torch.tensor([]).to(device)
            cnt += 1
    if cnt > 0:
        preds = get_pred(net, patch, torch.tensor(imgs), torch.tensor(metas))
        result_csv = process_preds(result_csv, rows, preds)
    result_csv.to_csv(split_path + "predictions_T" + str(TECHNIQUE) + "_" +str(MODEL) + "_" + name, index=False)
    
    # group = group.drop(drop_cols, axis=1)
    #     # print(group)
    #     i+=1
    #     if i>5:
    #         break
    #print(result_csv)


if __name__ == "__main__":
    generate_data("train_split.csv")
    generate_data("test_split.csv")
