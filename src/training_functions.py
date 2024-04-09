# This will be useful for training on CIFAR dataset

#important imports
# install torch from here https://pytorch.org/ 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
import sys
sys.path.append(os.path.abspath('..'))
from mammogram_dataset import MammogramDataset
from prediction_dataset import PredictionDataset
from pred_nn import PredNN, PredRNN
from progress_bar import progress_bar
from sklearn.metrics import balanced_accuracy_score, f1_score
import albumentations as A
import albumentations.pytorch as AP
import albumentations.augmentations.transforms as AT


import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="y_pred contains classes not in y_true")

def save_results(results, file_name):
    with open(file_name, 'w') as f:
        for result in results:
            f.write(str(result) + '\n')


def pfbeta(labels, predictions, beta = 1):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / max((ctp + cfp), 1)
    c_recall = ctp / max(y_true_count, 1)
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0


def get_dataset(batch_size, individual=False, get_cancer=True, tile=False, return_meta=False, split_path='../data_splits/standard/'):
    file = open(os.path.abspath(split_path+'mean_std.txt'), 'r')
    mean = float(file.readline())
    std = float(file.readline())
    file.close()
    
    transform_train = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.1, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.2, 0.2)),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        ], p=0.2),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, always_apply=True),
        A.augmentations.dropout.grid_dropout.GridDropout(p=0.15, random_offset=True, ratio=0.2),
        AP.ToTensorV2()
    ])
    transform_test = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, always_apply=True),
        AP.ToTensorV2()
    ])
    
    train_dataset = MammogramDataset(os.path.abspath(split_path+'balanced_train_split.csv'), 
                                     os.path.abspath('../processed_data/'), 
                                     transform=transform_train, 
                                     individual=individual, 
                                     get_cancer=get_cancer,
                                     tile=tile,
                                     return_meta=return_meta)
    
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    test_dataset = MammogramDataset(os.path.abspath(split_path+'test_split.csv'), 
                                    os.path.abspath('../processed_data/'), 
                                    transform=transform_test, 
                                    individual=individual, 
                                    get_cancer=get_cancer,
                                    tile=tile,
                                    return_meta=return_meta)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)   
    return train_dataloader, test_dataloader, train_dataset.get_bias()


def get_pred_dataset(batch_size, split_path='../data_splits/standard/', pred_type='T1_resnet50', ret_type="avg"):
    file_path_prefix = split_path + "balanced_predictions_" + pred_type + "_" 
    file_path_prefix_test = split_path + "predictions_" + pred_type + "_"
    
    train_dataset = PredictionDataset(os.path.abspath(file_path_prefix + 'train_split.csv'), ret_type=ret_type)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    
    test_dataset = PredictionDataset(os.path.abspath(file_path_prefix_test + 'test_split.csv'), ret_type=ret_type)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)   
    return train_dataloader, test_dataloader, 0


def get_model(model:str):
    if model == 'resnet18':
        result = torchvision.models.resnet18(num_classes=2)
        result.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        raise NotImplementedError
    elif model == 'resnet50':
        result = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        result.fc = torch.nn.Linear(2048, 2)
    elif model == 'resnext50':
        result = torchvision.models.resnext50_32x4d(num_classes=2)
        result.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        raise NotImplementedError
    elif model == 'vit':
        result = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        num_features = result.heads.head.in_features
        result.heads.head = torch.nn.Linear(num_features, 2)   
    elif model == 'pred_nn_avg':
        result = PredNN()
    elif model == 'pred_nn_amm':
        result = PredNN(input_dim=24)
    elif model == 'pred_nn_amms':
        result = PredNN(input_dim=25)
    elif model == 'pred_nn_pad':
        result = PredNN(input_dim=31)
    elif model == 'pred_rnn':
        result = PredRNN()
    else:
        assert False, "Model not supported"
    return result


def train(epoch, max_epochs, net, trainloader, optimizer, scheduler, criterion, device, cosine=False):
    net.train()
    train_loss = 0
    all_preds = []
    all_targets = []
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if not cosine:
            scheduler.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())
        progress_bar(epoch, max_epochs, batch_idx, len(trainloader), 'Loss: %.3f   Acc: %.3f   pF1: %3f'
                     % (train_loss/(batch_idx+1), 100.*balanced_accuracy_score(all_targets, all_preds), pfbeta(all_targets, all_preds)))
    if cosine:
        scheduler.step()


def test(epoch, max_epochs, net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

            progress_bar(epoch, max_epochs, batch_idx, len(testloader), 'Loss: %.3f   Acc: %.3f%%   pF1: %3f'
                         % (test_loss/(batch_idx+1), 100.*balanced_accuracy_score(all_targets, all_preds), pfbeta(all_targets, all_preds)))
    return balanced_accuracy_score(all_targets, all_preds)


def fit_model(model, trainloader, testloader, device, epochs:int, learning_rate:float, max_lr:float, momentum:float, save_path:str, bias=0.1, cosine=False):
    best_acc = -1
    best_name = ""
    weight = torch.tensor([bias, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=5e-4)
    if not cosine:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(trainloader))
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    for epoch in range(epochs):
        train(epoch, epochs, model, trainloader, optimizer, scheduler, criterion, device)
        acc = test(epoch, epochs, model, testloader, criterion, device)
        if acc > best_acc:
            if best_name != "":
                os.remove(best_name)
            best_acc = acc
            best_name = save_path + "_" + str(round(best_acc, 3)) + "_" + str(epoch) + ".pth" 
            torch.save(model.state_dict(), best_name)
    f = open(save_path + "_best.txt", "w")
    f.write(str(best_acc))
    f.close()
    return best_name, best_acc

