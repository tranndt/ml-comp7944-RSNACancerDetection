# Multiple Multi-Modal Methods of Malignant Mammogram Classification (M6C)

Before attempting to run the code, please install all requirements found in requirements.txt.  
Next, you will need to download the kaggle dataset for the scans (https://www.kaggle.com/competitions/rsna-breast-cancer-detection), or use the image versions of the scans posted for convinience (https://www.kaggle.com/datasets/radek1/rsna-mammography-images-as-pngs).  

Next, you will need to preprocess the data, which can be done by running data_preprocessing.py.  
Afterwards, splits need to be generated using generate_train_test_split.py and rebalanced with balance_dataset.py.  
Afterwards, the normalization values for each dataset can be generated using split_means_stds.py.  

Now that the data has been downloaded, preprocessed, split and balanced, it is ready to be trained on.  
Each relevant training code can be found in its relevant folder.  
Please note that we tried other techniques which we did not include in our paper as it ultimately did not fit our final theme.  
The techniques in the paper correspond to the folders in our project as follows:  
Naive classification (Method 1): technique_1/  
Destructive Patching (Method 2): technique_3/  
Early Concatenation (Method 3): technique_4/  
Bidirectional Cross Attention (Method 4): technique_4/  

To run each method, just use the train_model.py file found in each. The best performing models will automatically be saved in the folder, as well as a text file summarizing the training process.  Running these techniques requires substantial GPU memory and time.  

The code for stage 2 training can be found in pred_training/.  
In order to speed up training on the predictions made by various techniques, they were cached and turned into a new binary classification dataset, this was done using the generate_predicted_datasets.py, which accepts the names of the top performing models across each technique and will produce new datasets in the data_splits/ folder.  
Once the prediction datasets are produced, all classifiers, classifier parameters and Stage 2 methods can be grid searched on using sklearn_classifiers.py.  This will produce a textfile describing the top performing models and their configurations. 
Additionally in the pred_training/ folder, you can find methods for training RNNs, however this did not pan out and was not seriously brought up in the paper.  

Please let us know if you have any questions or would like our copy of the processed data.  
Here is out repo: https://github.com/tranndt/rsna-cancer-detection   

