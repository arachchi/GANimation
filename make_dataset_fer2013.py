import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

file = '/Users/JP25565/data/Kaggle/fer2013/fer2013.csv'
img_dir = 'fer2013/imgs'
scale_factor = 3


def main():

    # Make images
    df = pd.read_csv(file)

    if not os.path.exists(img_dir):
        os.makedirs(img_dir, exist_ok=True)

    df.pixels = df.pixels.map(lambda x: np.asarray(x.split(' '), dtype=np.float32).reshape(48, 48))
    df.pixels = df.pixels.map(
        lambda x: F.interpolate(
            torch.Tensor(x[np.newaxis, np.newaxis]), scale_factor=scale_factor, mode='bilinear', align_corners=False
        ).squeeze().numpy().astype(np.int32)
    )
    df.pixels = df.pixels.map(
        lambda x: x[:, :, np.newaxis].repeat(3, 2)
    )

    list_train = []
    list_test = []

    for item in tqdm(df.iterrows()):
        img_id, row = item
        img_name = '%05d_%s_%s.jpg' % (img_id, row.emotion, row.Usage)
        assert row.pixels.shape == (48*scale_factor, 48*scale_factor, 3)
        cv2.imwrite(os.path.join(img_dir, img_name), row.pixels)

        if row.Usage == 'Training':
            list_train.append(img_name)
        else:
            list_test.append(img_name)

    with open(os.path.join(img_dir, 'train_ids.csv').replace('/imgs', ''), 'w') as writer:
        writer.write('\n'.join(list_train))
    with open(os.path.join(img_dir, 'test_ids.csv').replace('/imgs', ''), 'w') as writer:
        writer.write('\n'.join(list_test))


def check_aus():
    import pickle
    pkl_file = './fer2013/aus.pkl'
    with open(pkl_file, 'rb') as reader:
        D = pickle.load(reader)
    for k, v in D.items():
        if v.shape[0] != 17:
            print(k)


if __name__ == '__main__':
    check_aus()
