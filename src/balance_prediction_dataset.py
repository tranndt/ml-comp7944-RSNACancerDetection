import pandas as pd
from sklearn.utils import resample
from itertools import count


def resample_data(path, technique, model):
    file_loc = path + "predictions_T" + str(technique) + "_" + model + "_train_split.csv"
    df = pd.read_csv(file_loc)
    df = df.dropna()
    class_counts = df['cancer'].value_counts()

    # Calculate the number of examples to match the majority class
    n_samples = class_counts.max()

    # Split the DataFrame into separate classes
    df_majority = df[df['cancer'] == 0]
    df_minority = df[df['cancer'] == 1]

    # Oversample the minority class to match the majority class
    df_minority_resampled = resample(df_minority, replace=True, n_samples=n_samples, random_state=42)
    maximum_patient_id = df_minority_resampled['patient_id'].max()
    print(maximum_patient_id)
    counter = count(start=maximum_patient_id + 1, step=1)
    df_minority_resampled['patient_id'] = df_minority_resampled.index.map(lambda x: next(counter))

    # Concatenate the resampled minority class with the majority class
    df_balanced = pd.concat([df_majority, df_minority_resampled])

    # Shuffle the DataFrame to mix the classes randomly
    # df_balanced = df_balanced.sample(frac=1, random_state=42)
    # class_counts_balanced = df_balanced['cancer'].value_counts()
    # print(class_counts_balanced)
    df_balanced.to_csv(path + 'balanced_predictions_T' + str(technique) + "_" + model + '_train_split.csv', index=False)


if __name__ == "__main__":
    resample_data('data_splits/standard/', 1, 'resnet50')
    resample_data('data_splits/standard/', 1, 'vit')
    resample_data('data_splits/standard/', 3, 'resnet50')
    resample_data('data_splits/standard/', 3, 'vit')
    resample_data('data_splits/standard/', 4, 'vit')
    
