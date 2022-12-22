import gdown
import argparse
import os

FILE_IDS = {
    "training.txt": "1SLot4nqFYth80UAG4lqRBH4eOWIDOTWO",
    "validation.txt": "1DQXFbCZhBDwOwdQ8BWSjs_UkAer5l3SB"
}

PRETRAINED_IDS = {
    "checkpoint.pth": "19ir-J80Uci7w09dyEI1JowPqItGDN2Js"
}


if __name__ == "__main__":
    models_dir = './pretrained_models/'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    
    for file_name, id in FILE_IDS.items():
        print(f'Loading {file_name}...')
        url = 'https://drive.google.com/uc?id=' + id
        gdown.download(url, file_name, quiet=True)
        
    for file_name, id in PRETRAINED_IDS.items():
        print(f'Loading {file_name}...')
        url = 'https://drive.google.com/uc?id=' + id
        gdown.download(url, models_dir + file_name, quiet=True)
    
    print('Done')