# This will be useful for training on CIFAR dataset

#important imports
# install torch from here https://pytorch.org/ 
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import time
import sys
sys.path.append(os.path.abspath('..'))
from training_functions import get_pred_dataset, get_model, save_results, pfbeta
from sklearn.metrics import balanced_accuracy_score, f1_score
from progress_bar import progress_bar
import random
import numpy as np
torch.random.manual_seed(0)
random.seed(0)
np.random.seed(0)


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


def train(epoch, max_epochs, net, trainloader, optimizer, scheduler, criterion, device, cosine=False):
    net.train()
    train_loss = 0
    all_preds = []
    all_targets = []
    
    for batch_idx, (inputs, imgs, count, targets) in enumerate(trainloader):
        inputs, imgs, targets = inputs.to(device), imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs, imgs, count)
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
        for batch_idx, (inputs, imgs, count, targets) in enumerate(testloader):
            inputs, imgs, targets = inputs.to(device), imgs.to(device), targets.to(device)
            outputs = net(inputs, imgs, count)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

            progress_bar(epoch, max_epochs, batch_idx, len(testloader), 'Loss: %.3f   Acc: %.3f%%   pF1: %3f'
                         % (test_loss/(batch_idx+1), 100.*balanced_accuracy_score(all_targets, all_preds), pfbeta(all_targets, all_preds)))
    return balanced_accuracy_score(all_targets, all_preds)


def main(dataset:str, pred_type:str, ret_type, model_name:str, epochs:int, learning_rate:float, batch_size:int, max_lr:float, momentum:float, output_prefix:str, cosine):
    print("CUDA Available: ", torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader, testloader, bias = get_pred_dataset(batch_size, ret_type=ret_type)
    model = get_model(model_name)
    model.to(device)
    os.makedirs("trained_rnn_models/" + pred_type + "/" + model_name +"/", exist_ok=True)
    best_name, best_accuracy = fit_model(model, trainloader, testloader, device, epochs, learning_rate, max_lr, momentum, "trained_rnn_models/" + pred_type + "/" + model_name + "/" + output_prefix + dataset + "_" + model_name, bias, cosine)
    print("Training complete: " + best_name + " with accuracy: " + str(round(best_accuracy, 4)))
    return str(round(best_accuracy, 4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('--dataset', type=str, default='mammograms', help='Dataset to train on')
    parser.add_argument('--model', type=str, default='pred_rnn', help='Model to train')
    parser.add_argument('--pred_type', type=str, default='T3_resnet50', help='The Technique and model type')
    parser.add_argument('--ret_type', type=str, default='rnn', help='Aggregation of predictions')
    parser.add_argument('--output_prefix', type=str, default='', help='Prefix to add to model name, to avoid overlapping experiments.')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD Momentum')
    parser.add_argument('--cosine', type=bool, default=True, help='Use Cosine Annealing')
    args = parser.parse_args()
    
    learning_rates = [4e-2, 3e-3, 1e-3]
    momentums = [0.9]
    cosines = [True]
    result_file = "results_" + str(time.time()) + ".txt"
    results = []
    
    
    for lr in learning_rates:
        for momentum in momentums:
            for cosine in cosines:
                print("Training with lr: " + str(lr) + " and momentum: " + str(momentum) + " cosine " + str(cosine))
                tag = "cosine_"+str(cosine)+"_"+str(lr)+"_"+str(momentum) 
                accuracy = main(args.dataset, args.pred_type, args.ret_type, args.model, args.epochs, lr, args.batch_size, args.max_lr, momentum, tag, cosine)
                results.append(tag + "___" + str(accuracy))
                save_results(results, result_file)
    
    
    #main(args.dataset, args.model, args.epochs, args.learning_rate, args.batch_size, args.max_lr, args.momentum, args.output_prefix, args.cosine)
