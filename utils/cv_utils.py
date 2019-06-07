import cv2
from matplotlib import pyplot as plt
import numpy as np

def read_cv2_img(path):
    '''
    Read images
    :param path: Path to image
    :return: 3-ch images (convert 1-ch to 3-ch)
    '''
    img = cv2.imread(path, -1)

    if img is not None:
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis].repeat(3, 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def show_cv2_img(img, title='img'):
    '''
    Display cv2 image
    :param img: cv::mat
    :param title: title
    :return: None
    '''
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_images_row(imgs, titles, rows=1):
    '''
       Display grid of cv2 images image
       :param img: list [cv::mat]
       :param title: titles
       :return: None
    '''
    assert ((titles is None) or (len(imgs) == len(titles)))
    num_images = len(imgs)

    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, num_images + 1)]

    fig = plt.figure()
    for n, (image, title) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(rows, np.ceil(num_images / float(rows)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        ax.set_title(title)
        plt.axis('off')
    plt.show()