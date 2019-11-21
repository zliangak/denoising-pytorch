import os
import tqdm
import copy
import numpy as np
import argparse

from utils import MISC, PublicTest, validation, test
from model import DA, ConvDa

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--patch_width', default=32, type=int)
parser.add_argument('--num_workers', default=2, type=str)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--sigma', default=10/255, type=float)
parser.add_argument('--wd', default=0, type=float)
parser.add_argument('--fake_train_length', default=50000, type=int)
parser.add_argument('--fake_val_length', default=10000, type=int)
parser.add_argument('--gpu', default='8', type=str)
parser.add_argument('--model', default='convda', type=str)
args = parser.parse_args()
print(args)

logs = {'train_loss':[],
        'val_loss':[],
        'test_loss': None,
        'test_psnr': None,
        'img_pairs': None
        }

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

data_root = '../data/'

train_transform = transforms.Compose([
    transforms.RandomCrop(args.patch_width),
    transforms.ToTensor()
     ])

test_transform = transforms.Compose([
    transforms.ToTensor()
     ])

dataset_train = MISC(os.path.join(data_root, 'train'), transform=train_transform, fake_length=args.fake_train_length)
dataset_val = MISC(os.path.join(data_root, 'val'), transform=train_transform, fake_length=args.fake_val_length)
dataset_test = PublicTest(os.path.join(data_root, 'test'), transform=test_transform, patch_width=args.patch_width)

train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)

if args.model == 'da':
    net = DA(patch_width=args.patch_width)
    net = net.cuda()
elif args.model == 'convda':
    net = ConvDa()
    net = net.cuda()
print(net)
print('num of parameters:', sum([p.numel() for p in net.parameters()]))

optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
criterion = torch.nn.MSELoss()
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 40], gamma=0.1)

for epoch in range(args.epochs):
    train_loss = 0
    for batch_idx, x in tqdm.tqdm(enumerate(train_loader), desc='Epoch %d'%epoch,
                                  total=int(args.fake_train_length//args.batch_size)):
        optimizer.zero_grad()

        x_noise = torch.clamp(copy.deepcopy(x) + torch.randn(x.shape) * args.sigma, 0, 1)
        x_noise = x_noise.cuda()
        x = x.cuda()
        output = net(x_noise)

        if args.model == 'da':
            loss = criterion(output, x.view(x.shape[0], -1))
        elif args.model == 'convda':
            loss = criterion(output, x)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    lr_scheduler.step()

    train_loss = train_loss / (batch_idx+1)
    val_loss = validation(net, val_loader, args.sigma, args)
    logs['train_loss'].append(train_loss)
    logs['val_loss'].append(val_loss)
    print('Train loss: {:5f} -- Val loss {:5f}'.format(train_loss, val_loss))

test_loss, test_psnr, img_pairs = test(net, test_loader, args.sigma, args.patch_width)
logs['test_loss'] = test_loss
logs['test_psnr'] = test_psnr
logs['img_pairs'] = img_pairs
print('Test loss: {:5f} -- Test PSNR {:5f}'.format(test_loss, test_psnr))

torch.save(logs, '../weights/logs_{}_epoch_{}_lr_{}_sigma_{:5f}_bs_{}_pw_{}_wd_{:5f}_valloss_{:5f}_testloss_{:5f}_testpnsr_{:5f}.pth'.format(
            args.model, args.epochs, args.lr, args.sigma, args.batch_size, args.patch_width, args.wd, val_loss, test_loss, test_psnr))
