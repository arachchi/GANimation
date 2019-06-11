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


def main():

    # Make images
    df = pd.read_csv(file)

    if not os.path.exists(img_dir):
        os.makedirs(img_dir, exist_ok=True)

    df.pixels = df.pixels.map(lambda x: np.asarray(x.split(' '), dtype=np.float32).reshape(48, 48))
    df.pixels = df.pixels.map(
        lambda x: F.interpolate(
            torch.Tensor(x[np.newaxis, np.newaxis]), scale_factor=3, mode='bilinear', align_corners=False
        ).squeeze().numpy().astype(np.int32)
    )

    list_train = []
    list_test = []

    for item in tqdm(df.iterrows()):
        img_id, row = item
        img_name = '%05d_%s_%s.png' % (img_id, row.emotion, row.Usage)
        cv2.imwrite(os.path.join(img_dir, img_name), row.pixels)

        if row.Usage == 'Training':
            list_train.append(img_name)
        else:
            list_test.append(img_name)

    with open(os.path.join(img_dir, 'train_ids.csv').replace('/imgs', ''), 'w') as writer:
        writer.write('\n'.join(list_train))
    with open(os.path.join(img_dir, 'test_ids.csv').replace('/imgs', ''), 'w') as writer:
        writer.write('\n'.join(list_test))


if __name__ == '__main__':
    main()
