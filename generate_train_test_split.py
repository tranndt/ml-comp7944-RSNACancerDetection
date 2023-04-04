import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

OUTPUT_LOC = "data_splits/"
SPLIT_RATIO = 0.1


def get_split(df, prefix="standard"):

    patient_ids = df.patient_id.unique()
    train_patient_ids, test_patient_ids = train_test_split(patient_ids, test_size=SPLIT_RATIO, random_state=42)

    # Convert categorical columns to integers using label encoding
    categorical_cols = ['laterality', 'view', 'cancer', 'biopsy', 'invasive', 'BIRADS', 'implant', 'density', 'machine_id', 'difficult_negative_case']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Normalize the age column
    df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()

    train_df = df[df['patient_id'].isin(train_patient_ids)]
    test_df = df[df['patient_id'].isin(test_patient_ids)]

    # Save the train and test sets to CSV files
    train_df.to_csv(OUTPUT_LOC + "/" + prefix + "/train_split.csv", index=False)
    test_df.to_csv(OUTPUT_LOC + "/" + prefix + "/test_split.csv", index=False)


if __name__ == "__main__":
    df = pd.read_csv("train.csv")
    get_split(df, "standard")
    cancer_difficult_df = df[(df['difficult_negative_case'] == 1) | (df['cancer'] == 1)]
    get_split(cancer_difficult_df, "hard")
    non_difficult_df = df[df['difficult_negative_case'] == 0]
    get_split(non_difficult_df, "easy")
