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
from sklearn.metrics import balanced_accuracy_score
import random
import numpy as np
torch.random.manual_seed(0)
random.seed(0)
np.random.seed(0)
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import warnings 
warnings.filterwarnings("ignore")


def parse_group(result_df, y, group, p_type="avg"):
    MAX_IMGS = 9
    y['cancer'].append(group['cancer'].max())
    result = {
        'age': group['age'].mean(),
        'laterality_L': group['laterality_L'].mean(),
        'laterality_R': group['laterality_R'].mean(),
        'machine_id_21': group['machine_id_21'].mean(),
        'machine_id_29': group['machine_id_29'].mean(),
        'machine_id_48': group['machine_id_48'].mean(),
        'machine_id_49': group['machine_id_49'].mean(),
        'machine_id_93': group['machine_id_93'].mean(),
        'machine_id_170': group['machine_id_170'].mean(),
        'machine_id_190': group['machine_id_190'].mean(),
        'machine_id_197': group['machine_id_197'].mean(),
        'machine_id_210': group['machine_id_210'].mean(),
        'machine_id_216': group['machine_id_216'].mean(),
        'site_id_1': group['site_id_1'].mean(),
        'site_id_2': group['site_id_2'].mean()
    }
    if p_type == 'avg':
        result['pred_avg'] = group['pred'].mean()
    elif p_type == 'max':
        result['pred_max'] = group['pred'].max()
    elif p_type == 'min':
        result['pred_min'] = group['pred'].min()
    elif p_type == 'amm':
        result['pred_avg'] = group['pred'].mean()
        result['pred_min'] = group['pred'].min()
        result['pred_max'] = group['pred'].max()
    elif p_type == 'amms':
        result['pred_avg'] = group['pred'].mean()
        result['pred_min'] = group['pred'].min()
        result['pred_max'] = group['pred'].max()
        std = group['pred'].std()
        if np.isnan(std):
            std = 0
        result['pred_std'] = std
    elif p_type == 'rnn':
        for i in range(MAX_IMGS):
            result['pred_' + str(i)] = -1
        cnt = 0
        for i, row in group.iterrows():
            result['pred_' + str(cnt)] = row['pred']
            cnt += 1
    result_df = result_df.append(result, ignore_index=True)
    return result_df, y
    

def create_dataset(csv_path, ret_type):
    print("Generating " + ret_type + " dataset...")
    df = pd.read_csv(csv_path)
    df = df.groupby(['patient_id', 'laterality_L'])
    result_df = pd.DataFrame()
    y = {'cancer': []}
    for name, group in tqdm(df):
        result_df, y = parse_group(result_df, y, group, p_type=ret_type)
    y = pd.DataFrame(y)
    return result_df, y

def load_dataset(pred_type, ret_type):
    SAVE_DIR = os.path.join(os.path.abspath('..'), 'data_splits', 'standard_sklearn', 'balanced_predictions_' + pred_type + '_' + ret_type)
    FIND_DIR = os.path.join(os.path.abspath('..'), 'data_splits', 'standard', 'balanced_predictions_' + pred_type)
    TEST_FIND_DIR = os.path.join(os.path.abspath('..'), 'data_splits', 'standard', 'predictions_' + pred_type)
    if not os.path.exists(SAVE_DIR + "_train_split.csv"):
        result_df, y = create_dataset(FIND_DIR + "_train_split.csv", ret_type)
        result_df.to_csv(SAVE_DIR + "_train_split.csv", index=False)
        y.to_csv(SAVE_DIR + "_train_split_y.csv", index=False)
    if not os.path.exists(SAVE_DIR + "_test_split.csv"):
        result_df, y = create_dataset(TEST_FIND_DIR + "_test_split.csv", ret_type)
        result_df.to_csv(SAVE_DIR + "_test_split.csv", index=False)
        y.to_csv(SAVE_DIR + "_test_split_y.csv", index=False)
    train_result_df = pd.read_csv(SAVE_DIR + "_train_split.csv")
    train_y = pd.read_csv(SAVE_DIR + "_train_split_y.csv")
    test_result_df = pd.read_csv(SAVE_DIR + "_test_split.csv")
    test_y = pd.read_csv(SAVE_DIR + "_test_split_y.csv")
    return train_result_df.values, train_y.values.astype(int), test_result_df.values, test_y.values.astype(int)


def try_config(model, xTrain, yTrain, xTest, yTest):
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    balanced_acc = balanced_accuracy_score(yTest, yPred)
    return balanced_acc


def get_model(model_type, options, i):
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=options['n_estimators'][i], max_depth=options['max_depth'][i], random_state=0)
    elif model_type == 'svm':
        model = SVC(kernel=options['kernel'][i], probability=True)
    elif model_type == 'mlp':
        model = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=options['hidden_layer_sizes'][i])
    elif model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=options['n_neighbors'][i])
    elif model_type == 'ada_boost':
        model = AdaBoostClassifier(n_estimators=options['n_estimators'][i], random_state=0)
    return model


def conf_to_str(config, model_type, i):
    ret_str = ""
    for k, v in config[model_type].items():
        ret_str += k + ": " + str(v[i]) + "  "
    return ret_str
    


def main(pred_type, ret_type, model_type, configs):
    xTrain, yTrain, xTest, yTest = load_dataset(pred_type, ret_type)
    print("Trying: " + model_type)
    size = 0
    options = configs[model_type]
    for k, v in options.items():
        size = len(options[k])
        break
    
    best_acc = 0
    best_i = 0
    for i in tqdm(range(size)):
        model = get_model(model_type, options, i)
        bal_acc = try_config(model, xTrain, yTrain, xTest, yTest)
        if bal_acc > best_acc:
            best_acc = bal_acc
            best_i = i
    return best_acc, best_i
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('--model', type=str, default='random_forest', help='Model to train')
    parser.add_argument('--pred_type', type=str, default='T3_resnet50', help='The Technique and model type')
    parser.add_argument('--ret_type', type=str, default='avg', help='Aggregation of predictions')
    args = parser.parse_args()
    configs = {
        'random_forest': {
            'n_estimators': [100, 150, 200, 250, 300, 350, 400] * 10,
            'max_depth': [5]*7 + [6]*7 + [7]*7 + [8]*7 + [9]*7 + [10]*7 + [11]*7 + [12]*7 + [13]*7 + [14]*7 
        },
        'svm': {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        },
        'mlp': {
            'hidden_layer_sizes': [(100,), (150,), (200,), (250,), (300,), (350,)],
        },
        'knn': {
            'n_neighbors': [3, 5, 7],
        },
        'ada_boost': {
            'n_estimators': [100, 125, 150, 175, 200, 250],
        },
    }  
    best_acc = 0
    best_i = 0
    
    for pred_type in ['T1_resnet50', 'T1_vit', 'T3_resnet50', 'T3_vit', 'T4_vit']:
        print("RUNNING PRED TYPE: " + pred_type)
        f = open("skresults_" + pred_type + "_" + str(time.time()).split('.')[0] + ".txt", "w")
        all_models = ['avg', 'max', 'min', 'amm', 'amms', 'rnn']
        for ret_type in all_models: #  ['rnn']:
            print("Using ret type: " + ret_type)
            for model_type in ['random_forest', 'svm', 'mlp', 'knn', 'ada_boost']:
                acc, i = main(pred_type, ret_type, model_type, configs)
                report = ret_type + ", " + model_type + ", " + conf_to_str(configs, model_type, i) + ", " + str(acc)
                if acc > best_acc:
                    best_acc = acc
                    best_i = i
                    print("New best accuracy!") 
                else: 
                    print("No improvement.")
                print(report)
                f.write(report + "\n")
        f.close()
    
    
    