import os
import shutil
from ctypes.wintypes import SIZE

from tqdm import tqdm

#To do this in code, you will probably need a function similar to this,
# where the original data frame has entries of the images, their class id, and their bounding boxes:
def create_file(df, split_df, train_file, train_folder, fold):
    os.makedirs('labels/train/', exist_ok=True)
    os.makedirs('images/train/', exist_ok=True)
    os.makedirs('labels/val/', exist_ok=True)
    os.makedirs('images/val/', exist_ok=True)

    list_image_train = split_df[split_df[f'fold_{fold}'] == 0]['image_id']
    train_df = df[df['image_id'].isin(list_image_train)].reset_index(drop=True)
    val_df = df[~df['image_id'].isin(list_image_train)].reset_index(drop=True)

    for train_img in tqdm(train_df.image_id.unique()):
        with open('labels/train/{train_img}.txt', 'w+') as f:
            row = train_df[train_df['image_id'] == train_img] \
                [['class_id', 'x_center', 'y_center', 'width', 'height']].values
            row[:, 1:] /= SIZE  # Image size, 512 here
            row = row.astype('str')
            for box in range(len(row)):
                text = ' '.join(row[box])
                f.write(text)
                f.write('\n')
        shutil.copy(f'{train_img}.png',
                    f'images/train/{train_img}.png')
'''
    for val_img in tqdm(val_df.image_id.unique()):
        with open(f'{labels / val / {val_img}.txt', 'w+') as f:
            row = val_df[val_df['image_id'] == val_img] \
                [['class_id', 'x_center', 'y_center', 'width', 'height']].values
            row[:, 1:] /= SIZE
            row = row.astype('str')
            for box in range(len(row)):
                text = ' '.join(row[box])
                f.write(text)
                f.write('\n')
        shutil.copy(f'{val_img}.png',
                    f'images/val/{val_img}.png')'''


#you need a config file with the names of the labels, the number of classes, and the training & validation paths.
'''
import yaml
classes = [ ‘Aortic enlargement’,
 ‘Atelectasis’,
 ‘Calcification’,
 ‘Cardiomegaly’,
 ‘Consolidation’,
 ‘ILD’,
 ‘Infiltration’,
 ‘Lung Opacity’,
 ‘Nodule/Mass’,
 ‘Other lesion’,
 ‘Pleural effusion’,
 ‘Pleural thickening’,
 ‘Pneumothorax’,
 ‘Pulmonary fibrosis’]
data = dict(
 train = ‘../vinbigdata/images/train’, # training images path
 val = ‘../vinbigdata/images/val’, # validation images path
 nc = 14, # number of classes
 names = classes
 )
with open(‘./yolov5/vinbigdata.yaml’, ‘w’) as outfile:
 yaml.dump(data, outfile, default_flow_style=False)'''