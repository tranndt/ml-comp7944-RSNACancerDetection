"""
Reduce dataset size! Convert all DICOM images to PNGs and resize them to 800x800.    
"""

import os
import pydicom
from PIL import Image
from tqdm import tqdm
import numpy as np


DATA_DIR = "/media/chris/Shared Disk/rsna-breast-cancer-detection/train_images"
OUTPUT_DIR = "/home/chris/Documents/Projects/rsna-cancer-detection/processed_data"
PRE_CROP_SIZE = 512


def get_pngs(data_dir, output_dir=None, crop_size=500):
    if output_dir is None:
        output_dir = data_dir + "_png/"
    else:
        output_dir += "/"
    data_dir += "/"    
    
    dirs = [d for d in os.listdir(data_dir)]

    for directory in tqdm(dirs):
        cur_dur = os.path.join(data_dir, directory)
        out_dir = os.path.join(output_dir, directory)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for file in os.listdir(cur_dur):
            file_path = os.path.join(cur_dur, file)
            ds = pydicom.dcmread(file_path)
            image = ds.pixel_array            
            image = image.astype(float)  
            image = image / image.max()
            if ds.PhotometricInterpretation == "MONOCHROME1":
                image = 1 - image
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image).resize((crop_size, crop_size))
            image.save(os.path.join(out_dir, file.replace(".dcm", ".png")))
        

if __name__ == "__main__":
    get_pngs(DATA_DIR, OUTPUT_DIR, PRE_CROP_SIZE)
    