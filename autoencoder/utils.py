import os
import copy
import numpy as np
import copy

from PIL import Image

from torch.utils.data import Dataset
import torch

class MISC(Dataset):
    '''process the training/validaiton set'''
    def __init__(self, root, transform, fake_length=50000):
        ls = os.listdir(root)
        self.transform = transform
        self.imgs = []
        for i, l in enumerate(ls):
            img = Image.open(os.path.join(root,l))
            self.imgs.append(img)
        self.real_length = len(self.imgs)
        self.fake_length = fake_length

    def __len__(self):
        '''since we only have a few images, so we
        fool the dataloder by sampling the images with replacement when
        creating a batch'''
        return self.fake_length

    def __getitem__(self, idx):
        return self.transform(self.imgs[idx%self.real_length])

class PublicTest(Dataset):
    '''process the public test set'''
    def __init__(self, root, transform, patch_width):
        ls = os.listdir(root)
        self.transform = transform
        self.pw = patch_width
        self.imgs = {}
        for i, l in enumerate(ls):
            img = Image.open(os.path.join(root,l))
            self.imgs[l.split('.')[0]] = img
        self.keys = list(self.imgs.keys())

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[self.keys[idx]]
        return self.transform(img)


def validation(net, loader, sigma, args):
    '''calculate the validation loss'''
    criterion = torch.nn.MSELoss()
    net.eval()
    total_loss = 0

    for batch_idx, x in enumerate(loader):

        x_noise = torch.clamp(copy.deepcopy(x) + torch.randn(x.shape) * sigma, 0, 1)
        x_noise = x_noise.cuda()
        x = x.cuda()
        output = net(x_noise)

        if args.model == 'da':
            loss = criterion(output, x.view(x.shape[0], -1))
        elif args.model == 'convda':
            loss = criterion(output, x)

        total_loss += loss.item()

    return total_loss/(batch_idx+1)


def get_patches(img, patch_width):
    '''split the testing image into patch with specific width'''
    pw = patch_width
    h, w = img.shape
    patches = torch.zeros((int(h // pw), int(w // pw), pw, pw))
    for h_idx in range(int(h // pw)):
        for w_idx in range(int(w // pw)):
            patches[h_idx, w_idx, :, :] = img[h_idx * pw:(h_idx + 1) * pw, w_idx * pw:(w_idx + 1) * pw]
    return patches


def calculate_mse(img1, img2):
    return (img1-img2).pow(2).sum() / img1.numel()


def calculate_psnr(img1, img2):
    MAX = 1
    return 10 * torch.log(MAX**2/calculate_mse(img1, img2)) / torch.log(torch.tensor(10.))


def test(net, loader, sigma, patch_width):
    '''using public test image for testing
    image can be downloaded from:
    https://www.io.csic.es/PagsPers/JPortilla/image-processing/bls-gsm/63-test-images
    '''
    net.eval()
    pw = patch_width
    total_mse = []
    total_psnr = []
    img_pairs = []
    for batch_idx, img1 in enumerate(loader):
        img1 = copy.deepcopy(img1[0,0]) # original image
        img_noise = torch.clamp(copy.deepcopy(img1) + torch.randn(img1.shape) * sigma, 0, 1) # noise image
        patches = get_patches(img_noise, pw)
        img2 = torch.zeros_like(img1) # recovered image
        h, w = img1.shape
        for h_idx in range(int(h//pw)):
            for w_idx in range(int(w//pw)):
                patch = patches[h_idx, w_idx,:, :]
                patch = patch[None, None, :, :]
                patch = patch.cuda()
                patch_clean = net(patch)
                img2[h_idx * pw:(h_idx + 1) * pw, w_idx * pw:(w_idx + 1) * pw] = patch_clean.view(pw, pw).detach()
        mse = calculate_mse(img1, img2)
        psnr = calculate_psnr(img1, img2)
        total_mse.append(mse.item())
        total_psnr.append(psnr.item())
        img_pairs.append([copy.deepcopy(img1.numpy()), copy.deepcopy(img_noise.numpy()), copy.deepcopy(img2.numpy())])

    return np.mean(total_mse), np.mean(total_psnr), img_pairs


