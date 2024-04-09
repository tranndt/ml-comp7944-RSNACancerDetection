import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_stats(path):
    df = pd.read_csv(path + 'train_split.csv')
    pixels = 0
    mean = 0
    M2 = 0
    count = 0

    for index, row in tqdm(df.iterrows()):
        # Load the image from the 'path' column
        img = Image.open('processed_data/' + str(int(row['patient_id'])) + '_' + str(int(row['image_id'])) + '.png').convert('L')
        img = img.resize((512, 512))
        # Convert the image to a NumPy array
        img_data = np.array(img)
        
        # Update running mean and M2
        pixels = np.prod(img_data.shape)
        delta = img_data - mean
        mean += np.sum(delta) / pixels
        M2 += np.sum(delta * (img_data - mean))
        count += pixels

    variance = M2 / count
    std = np.sqrt(variance)
    mean = int(mean)
    std = int(std)
    with open(path + 'mean_std.txt', 'w') as f:
        f.write(str(mean) + '\n' + str(std))


if __name__ == "__main__":
    get_stats('data_splits/standard/')
    get_stats('data_splits/hard/')
