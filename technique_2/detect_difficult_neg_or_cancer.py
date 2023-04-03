# This will be useful for training on CIFAR dataset

#important imports
# install torch from here https://pytorch.org/ 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import sys
sys.path.append(os.path.abspath('..'))
from mammogram_dataset import MammogramDataset
from progress_bar import progress_bar


def get_dataset(batch_size, individual=False, get_cancer=True):
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])])
    train_dataset = MammogramDataset(os.path.abspath('../data_splits/standard/train_split.csv'), os.path.abspath('../processed_data/'), transform=transform, individual=individual, get_cancer=get_cancer)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataset = MammogramDataset(os.path.abspath('../data_splits/standard/test_split.csv'), os.path.abspath('../processed_data/'), transform=transform, individual=individual, get_cancer=get_cancer)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)   
    return train_dataloader, test_dataloader


def get_model(model:str):
    if model == 'resnet18':
        result = torchvision.models.resnet18(num_classes=2)
        result.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif model == 'resnet50':
        result = torchvision.models.resnet50(num_classes=2)
        result.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        assert False, "Model not supported"
    return result


def train(epoch, max_epochs, net, trainloader, optimizer, scheduler, criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(epoch, max_epochs, batch_idx, len(trainloader), 'Loss: %.3f   Acc: %.3f%%'
                     % (train_loss/(batch_idx+1), 100.*correct/total))
    # scheduler.step()


def test(epoch, max_epochs, net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(epoch, max_epochs, batch_idx, len(testloader), 'Loss: %.3f   Acc: %.3f%%'
                         % (test_loss/(batch_idx+1), 100.*correct/total))
    return float(correct)/total


def fit_model(model, trainloader, testloader, device, epochs:int, learning_rate:float, max_lr:float, momentum:float, save_path:str):
    best_acc = -1
    best_name = ""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(trainloader))
    
    for epoch in range(epochs):
        train(epoch, epochs, model, trainloader, optimizer, scheduler, criterion, device)
        acc = test(epoch, epochs, model, testloader, criterion, device)
        if acc > best_acc:
            if best_name != "":
                os.remove(best_name)
            best_acc = acc
            best_name = save_path + "_" + str(epoch) + ".pth"
            torch.save(model.state_dict(), best_name)
    f = open(save_path + "_best.txt", "w")
    f.write(str(best_acc))
    f.close()
    return best_name, best_acc


def main(dataset:str, model_name:str, epochs:int, learning_rate:float, batch_size:int, max_lr:float, momentum:float, output_prefix:str):
    print("CUDA Available: ", torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader, testloader = get_dataset(batch_size, individual=True, get_cancer=False)    
    model = get_model(model_name)
    model.to(device)
    os.makedirs("trained_models_difficult/" + model_name +"/", exist_ok=True)
    best_name, best_accuracy = fit_model(model, trainloader, testloader, device, epochs, learning_rate, max_lr, momentum, "trained_models_difficult/" + model_name + "/" + output_prefix + dataset + "_" + model_name)
    print("Training complete: " + best_name + " with accuracy: " + str(round(best_accuracy, 4)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('--dataset', type=str, default='mammograms', help='Dataset to train on')
    parser.add_argument('--model', type=str, default='resnet50', help='Model to train')
    parser.add_argument('--output_prefix', type=str, default='', help='Prefix to add to model name, to avoid overlapping experiments.')
    parser.add_argument('--epochs', type=int, default=120, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD Momentum')
    args = parser.parse_args()
    main(args.dataset, args.model, args.epochs, args.learning_rate, args.batch_size, args.max_lr, args.momentum, args.output_prefix)
