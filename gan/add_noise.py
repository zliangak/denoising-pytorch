import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from skimage.util import random_noise

####transfer pgm to jpg
load_path = './data/pgm'
save_path = './data/jpg'
listpath = os.listdir(load_path)
for filename in np.sort(listpath):
    im = Image.open(os.path.join(load_path, filename))
    filename_new = filename[:-4] + '.jpg'
    im.save(os.path.join(save_path, filename_new))


#### add gaussian noise
clean_img = []
noisy_img = []
SNR = 1   # signal noise ratio
num = 200
listpath = os.listdir(save_path)
for filename in np.sort(listpath):
    im = plt.imread(os.path.join(save_path, filename))
    im = rescale_intensity(1.0 * im, out_range=(0, 1))
    print(im.shape)
    for i in range(num):
        clean_img.append(im)
        noisy_img.append(random_noise(im, mode='gaussian', var=im.var() / SNR))   ###add gaussian noise, here SNR = 1
np.save('./data/clean_img.npy', clean_img)
np.save('./data/noisy_img_SNR' + str(SNR) + '.npy', noisy_img)
