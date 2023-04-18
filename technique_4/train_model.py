# This will be useful for training on CIFAR dataset

#important imports
# install torch from here https://pytorch.org/ 
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import sys
sys.path.append(os.path.abspath('..'))
from patch_producer import PatchProducer
import time
from progress_bar import progress_bar
from sklearn.metrics import balanced_accuracy_score, f1_score
from training_functions import get_dataset, get_model, save_results, pfbeta
from CustomVIT import vit_b_16
import random
import numpy as np
torch.random.manual_seed(0)
random.seed(0)
np.random.seed(0)


def train(epoch, max_epochs, net, patch_producer, trainloader, optimizer, scheduler, criterion, device, cosine=False):
    net.train()
    patch_producer.train()
    train_loss = 0
    all_preds = []
    all_targets = []
    
    for batch_idx, (inputs, targets, meta) in enumerate(trainloader):
        inputs, targets, meta = inputs.to(device), targets.to(device), meta.to(device)
        optimizer.zero_grad()
        patch = patch_producer(meta)
        patch = patch.reshape(patch.shape[0], -1)
        outputs = net(inputs, patch)
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


def test(epoch, max_epochs, net, patch_producer, testloader, criterion, device):
    net.eval()
    patch_producer.eval()
    test_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, meta) in enumerate(testloader):
            inputs, targets, meta = inputs.to(device), targets.to(device), meta.to(device)
            patch = patch_producer(meta)
            patch = patch.reshape(patch.shape[0], -1)
            outputs = net(inputs, patch)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

            progress_bar(epoch, max_epochs, batch_idx, len(testloader), 'Loss: %.3f   Acc: %.3f%%   pF1: %3f'
                         % (test_loss/(batch_idx+1), 100.*balanced_accuracy_score(all_targets, all_preds), pfbeta(all_targets, all_preds)))
    return balanced_accuracy_score(all_targets, all_preds)

def fit_model(model, patch_producer, trainloader, testloader, device, epochs:int, learning_rate:float, lr_p, max_lr:float, momentum:float, save_path:str, bias=0.1, cosine=False):
    best_acc = -1
    best_name = ""
    
    # weight = torch.tensor([bias, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': model.parameters(), 'weight_decay': 5e-4, 'lr': learning_rate, 'momentum': momentum},
        {'params': patch_producer.parameters(), 'weight_decay': 5e-4, 'lr': lr_p, 'momentum': momentum}])
    if not cosine:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(trainloader))
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    for epoch in range(epochs):
        train(epoch, epochs, model, patch_producer, trainloader, optimizer, scheduler, criterion, device)
        acc = test(epoch, epochs, model, patch_producer, testloader, criterion, device)
        if acc > best_acc:
            if best_name != "":
                os.remove(best_name)
            best_acc = acc
            best_name = save_path + "_" + str(round(best_acc, 3)) + "_" + str(epoch) + ".pth" 
            best_name_patch = save_path + "patch_" + str(round(best_acc, 3)) + "_" + str(epoch) + ".pth" 
            torch.save(model.state_dict(), best_name)
            torch.save(patch_producer.state_dict(), best_name_patch)
    f = open(save_path + "_best.txt", "w")
    f.write(str(best_acc))
    f.close()
    return best_name, best_acc


def main(dataset:str, model_name:str, epochs:int, learning_rate:float, lr_p, batch_size:int, max_lr:float, momentum:float, output_prefix:str, cosine:bool):
    print("CUDA Available: ", torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader, testloader, bias = get_dataset(batch_size, individual=True, return_meta=True, tile=True)    
    model = get_model(model_name)
    model = vit_b_16(intermediate_embedding_size=768)
    model.to(device)
    patch_producer = PatchProducer()
    model.to(device)
    patch_producer.to(device)
    os.makedirs("trained_models/" + model_name +"/", exist_ok=True)
    best_name, best_accuracy = fit_model(model, patch_producer, trainloader, testloader, device, epochs, learning_rate, lr_p, max_lr, momentum, "trained_models/" + model_name + "/" + output_prefix + dataset + "_" + model_name, bias, cosine)
    print("Training complete: " + best_name + " with accuracy: " + str(round(best_accuracy, 4)))
    return str(round(best_accuracy, 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('--dataset', type=str, default='mammograms', help='Dataset to train on')
    parser.add_argument('--model', type=str, default='vit', help='Model to train')
    parser.add_argument('--output_prefix', type=str, default='', help='Prefix to add to model name, to avoid overlapping experiments.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--learning_rate_p', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--momentum', type=float, default=0.4, help='SGD Momentum')
    parser.add_argument('--cosine', type=bool, default=True, help='Use Cosine Annealing')
    args = parser.parse_args()
    
    learning_rates = [1e-3,5e-4]
    learning_rates_p = [1e-4,5e-4]
    momentums = [0.7,0.9]
    cosines = [True]
    result_file = "results_" + str(time.time()) + ".txt"
    results = []
    
    for lr in learning_rates:
        for lr_p in learning_rates_p:
            for momentum in momentums:
                for cosine in cosines:
                    print("Training with lr: " + str(lr) + "and lrp " + str(lr_p) + " and momentum: " + str(momentum) + " cosine " + str(cosine))
                    tag = "cosine_"+str(cosine)+"_"+str(lr)+"_"+str(lr_p)+"_"+str(momentum) 
                    accuracy = main(args.dataset, args.model, args.epochs, lr, lr_p, args.batch_size, args.max_lr, momentum, tag, cosine)
                    results.append(tag + "___" + str(accuracy))
                    save_results(results, result_file)


