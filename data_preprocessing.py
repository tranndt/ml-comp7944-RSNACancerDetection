import sys
import os
sys.path.append(os.path.abspath("GMIC/"))
from GMIC.src.cropping.crop_mammogram import crop_mammogram_one_image
from tqdm import tqdm
import pandas as pd
from PIL import Image
import cv2


DATA_DIR = "data/output"
OUTPUT_DIR = "processed_data"
PRE_CROP_SIZE = 1024
POST_CROP_SIZE = 512


def denoise(img, strength=4, window_size=11, window_search=22):
    return cv2.fastNlMeansDenoising(img, None, strength, window_size, window_search) 


def clahe(img, clip_limit=4.5, tile_size=(32,32)):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_enhanced = clahe.apply(gray)
    return image_enhanced


def bilateral_filtering(img):
    return cv2.bilateralFilter(img, 5, 20, 20)


def make_square_resize(input_path, expand):
    img = cv2.imread(input_path)
    height, width, _ = img.shape
    size = max(height, width)
    top, bottom, left, right = 0, 0, 0, 0

    if expand == 'L':
        left = size - width
    else:
        right = size - width

    color = [0, 0, 0]  # black color
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    img = cv2.resize(img, (size, size))

    img = clahe(img)
    img = bilateral_filtering(img)
    img = denoise(img)


    cv2.imwrite(input_path, img)


def get_pngs(data_dir, output_dir=None):
    if output_dir is None:
        output_dir = data_dir + "_png/"
    else:
        output_dir += "/"
    data_dir += "/"    
    df = pd.read_csv("train.csv")
    print(df)
    dirs = [d for d in os.listdir(data_dir)]

    for file in tqdm(dirs):
        image_id = int(file.split(".")[0].split("_")[1])
        row = df.loc[df["image_id"] == image_id]
        meta = {
            'image_id': image_id,
            'horizontal_flip': 'NO',
            'side': row["laterality"].values[0],
        }
        crop_mammogram_one_image(meta, os.path.abspath(os.path.join(data_dir, file)), os.path.abspath(os.path.join(output_dir, file)), 50, 15)
        make_square_resize(os.path.abspath(os.path.join(output_dir, file)), meta['side'])
        
        

if __name__ == "__main__":
    get_pngs(DATA_DIR, OUTPUT_DIR)
    
