# This will be useful for training on CIFAR dataset

#important imports
# install torch from here https://pytorch.org/ 
import torch
import os
import argparse
import time
import sys
sys.path.append(os.path.abspath('..'))
from training_functions import get_pred_dataset, get_model, fit_model, save_results
import random
import numpy as np
torch.random.manual_seed(0)
random.seed(0)
np.random.seed(0)


def main(dataset:str, pred_type:str, ret_type, model_name:str, epochs:int, learning_rate:float, batch_size:int, max_lr:float, momentum:float, output_prefix:str, cosine):
    print("CUDA Available: ", torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader, testloader, bias = get_pred_dataset(batch_size, ret_type=ret_type)
    model = get_model(model_name)
    model.to(device)
    os.makedirs("trained_models/" + pred_type + "/" + model_name +"/", exist_ok=True)
    best_name, best_accuracy = fit_model(model, trainloader, testloader, device, epochs, learning_rate, max_lr, momentum, "trained_models/" + pred_type + "/" + model_name + "/" + output_prefix + dataset + "_" + model_name, bias, cosine)
    print("Training complete: " + best_name + " with accuracy: " + str(round(best_accuracy, 4)))
    return str(round(best_accuracy, 4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('--dataset', type=str, default='mammograms', help='Dataset to train on')
    parser.add_argument('--model', type=str, default='pred_nn_avg', help='Model to train')
    parser.add_argument('--pred_type', type=str, default='T3_resnet50', help='The Technique and model type')
    parser.add_argument('--ret_type', type=str, default='avg', help='Aggregation of predictions')
    parser.add_argument('--output_prefix', type=str, default='', help='Prefix to add to model name, to avoid overlapping experiments.')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD Momentum')
    parser.add_argument('--cosine', type=bool, default=True, help='Use Cosine Annealing')
    args = parser.parse_args()
    
    learning_rates = [1e-2]
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
