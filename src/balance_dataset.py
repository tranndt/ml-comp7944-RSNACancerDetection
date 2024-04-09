import pandas as pd
from sklearn.utils import resample


def resample_data(path):
    df = pd.read_csv(path + 'train_split.csv')
    df = df.dropna()
    class_counts = df['cancer'].value_counts()

    # Calculate the number of examples to match the majority class
    n_samples = class_counts.max()

    # Split the DataFrame into separate classes
    df_majority = df[df['cancer'] == 0]
    df_minority = df[df['cancer'] == 1]

    # Oversample the minority class to match the majority class
    df_minority_resampled = resample(df_minority, replace=True, n_samples=n_samples, random_state=42)

    # Concatenate the resampled minority class with the majority class
    df_balanced = pd.concat([df_majority, df_minority_resampled])

    # Shuffle the DataFrame to mix the classes randomly
    df_balanced = df_balanced.sample(frac=1, random_state=42)
    class_counts_balanced = df_balanced['cancer'].value_counts()
    print(class_counts_balanced)
    df_balanced.to_csv(path + 'balanced_train_split.csv', index=False)


if __name__ == "__main__":
    resample_data('data_splits/standard/')
    resample_data('data_splits/hard/')
