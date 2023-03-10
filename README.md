# rsna-cancer-detection


Project Statement


Implementation 
- Data Preprocessing
    - Crop out black background from images
    - Normalize images to the same size 
    - Reduce details of images

- Architecture
    - Have (regular) models learn and produce either score or embeddings from each picture -> Feed into a final model + patient's other info -> Final prediction

- Dataset EDA
- https://www.kaggle.com/code/andradaolteanu/rsna-breast-cancer-eda-pytorch-baseline
- Overview 
    - Target
        - Implants: 1477 - 53229 (2)
        - Cancer (1/0): 1158-53548 (2%)
        - Invasive cancer (T/F): 818-340 (70%)
        - Diffucult cases (T/F): 7705-47001 (16%)
    - Train preprocessing
        - Laterality: L - R
        - View: 0 - 5


Image
[https://www.kaggle.com/code/theoviel/dicom-resized-png-jpg](Dicom -> Resized PNG/JPG)
    - Viewing images using pydicom
    - Resizing images

        - 256x256 : https://www.kaggle.com/datasets/theoviel/rsna-breast-cancer-256-pngs

        - 512x512 : https://www.kaggle.com/datasets/theoviel/rsna-breast-cancer-512-pngs

        - 1024x1204: https://www.kaggle.com/datasets/theoviel/rsna-breast-cancer-1024-pngs
        

[Cropping images and scaling methods](https://www.kaggle.com/code/chg0901/new-crop-and-hist-scaled-method-with-dali-tensor)
- Crop, flip and normalize values


[RSNA: Cut Off Empty Space from Images](https://www.kaggle.com/code/vslaykovsky/rsna-cut-off-empty-space-from-images)

Models:
- Fast AI
- ResNet 50
