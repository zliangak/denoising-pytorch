import os, time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import criterion
from model import dcgan
import numpy as np
import argparse
import math

parser = argparse.ArgumentParser()

parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train')
parser.add_argument('--lrd', type=float, default=0.001, help='learning rate of discriminator')
parser.add_argument('--lrg', type=float, default=0.01, help='learning rate of generator')
parser.add_argument('--device', type=str, default='cuda', help='device assignment')
parser.add_argument('--arch', type=str, default='wgan', help='network architecture')
parser.add_argument('--test_name', type=str, default='test', help='name of test')
parser.add_argument('--train_name', type=str, default='train', help='name of train')
parser.add_argument('--var', type=float, default=10, help='variance of noise equal (var/255)**2')
parser.add_argument('--train_size', type=int, default=6000, help='size of training samples')

opt = parser.parse_args()

# training parameters
batch_size = opt.batchSize
lrd = opt.lrd
lrg = opt.lrg
var = opt.var
mname = opt.arch + 'var' + str(var)
train_epoch = opt.nepoch
size = opt.train_size
img_size = opt.imageSize
test_name = opt.test_name
train_name = opt.train_name

USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device(opt.device)
else:
    device = torch.device(opt.device)

# set folder
if not os.path.isdir('./save_net'):
    os.mkdir('./save_net')
if not os.path.isdir('./save_net/' + mname):
    os.mkdir('./save_net/' + mname)
if not os.path.isdir('./save_img'):
    os.mkdir('./save_img')
if not os.path.isdir('./save_img/' + mname):
    os.mkdir('./save_img/' + mname)
if not os.path.isdir('./temp_img'):
    os.mkdir('./temp_img')


###train
def train():
    ###network
    G = dcgan.generator()
    D = dcgan.discriminator()
    BCE_loss = nn.BCEWithLogitsLoss()  # Binary Cross Entropy loss
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G_optimizer = optim.Adam(G.parameters(), lr=lrg, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lrd, betas=(0.5, 0.999))
    G.cuda()
    D.cuda()

    ##data
    noisy_data = np.load('./data/' + train_name + '_noisy_img_var' + str(var) + '.npy')
    clean_data = np.load('./data/' + train_name + '_clean_img.npy')
    D_loss = []
    G_loss = []
    for epoch in range(train_epoch):
        k = 0
        epoch_start_time = time.time()
        for i in range(int(6000 / batch_size)):
            cond = noisy_data[i * batch_size: i * batch_size + batch_size, :, :]
            img = clean_data[i * batch_size: i * batch_size + batch_size, :, :]
            k = k + 1
            pimg2 = np.reshape(img, (batch_size, 1, img_size, img_size))
            pcond2 = np.reshape(cond, (batch_size, 1, img_size, img_size))
            pimg2 = torch.from_numpy(pimg2)
            pcond2 = torch.from_numpy(pcond2)
            pimg2 = pimg2.to(device=device, dtype=dtype)
            pcond2 = pcond2.to(device=device, dtype=dtype)

            ##wgan
            ###d_step
            D.zero_grad()
            D_result = D(pimg2)
            D_result = D_result.squeeze()

            G_result = G(pcond2)

            D_train_loss = -torch.mean(D_result) + torch.mean(D(G_result))

            D_train_loss.backward()
            D_optimizer.step()

            # D_losses.append(D_train_loss.data[0])
            D_loss.append(D_train_loss.item())

            # Clip weights of discriminator
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

            #  G_step

            G.zero_grad()
            G_result = G(pcond2)
            D_result = D(G_result).squeeze()
            G_train_loss = -torch.mean(D_result) + 100 * torch.mean(torch.abs(G_result - pimg2))
            G_train_loss.backward(retain_graph=True)
            G_optimizer.step()

            # jsgan
            '''y_real_ = torch.ones((batch_size, 13, 13))
            y_fake_ = torch.zeros((batch_size, 13, 13))
            y_real_ = y_real_.to(device=device, dtype=dtype)
            y_fake_ = y_fake_.to(device=device, dtype=dtype)
            if k % 2 == 0:
                ###d_step
                D.zero_grad()
                D_result = D(pimg2)
                D_result = D_result.squeeze()
                D_real_loss = BCE_loss(D_result, y_real_)
                G_result = G(pcond2)
                D_result = D(G_result).squeeze()
                D_fake_loss = BCE_loss(D_result, y_fake_)
                D_train_loss = D_real_loss + D_fake_loss
                D_train_loss.backward()
                D_optimizer.step()
                D_loss.append(D_train_loss.item())

            #  G_step
            G.zero_grad()
            G_result = G(pcond2)
            D_result = D(G_result).squeeze()
            G_train_loss = BCE_loss(D_result, y_real_) + 100 * torch.mean(torch.abs(G_result - pimg2))
            G_train_loss.backward(retain_graph=True)
            G_optimizer.step()'''
            G_loss.append(G_train_loss.item())

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            if k % 50 == 0:
                print(
                    '[%d/%d] - lrd:%.5f, lrg:%.5f, lgptime: %.2f, loss_d: %.3f, loss_g: %.3f, l1loss: %.3f, l2loss: %.5f, discriminator_loss: %.3f' % (
                        k, epoch, lrd, lrg, per_epoch_ptime, D_train_loss.item(),
                        G_train_loss.item(), 100 * torch.mean(torch.abs(G_result - pimg2)),
                        torch.mean((G_result - pimg2) ** 2), torch.mean(F.sigmoid(D_result))))
            mse = []
            psnr = []
            if k % 200 == 0:
                for t in range(batch_size):
                    a = G_result[t].cpu().detach().numpy()
                    a = np.reshape(a, (img_size, img_size))
                    m = criterion.MSE(a, img[t])
                    print(m)
                    mse.append(m)
                    psnr.append(10 * math.log(1 / m))
                plt.imshow(a, cmap='gray')
                plt.savefig('./temp_img/' + str(epoch) + '_' + str(k / 200) + '.png')
                print('mse:%.6f' % np.mean(mse))
        np.save('./save_net/' + mname + '/train_mse.npy', mse)
        np.save('./save_net/' + mname + '/train_psnr.npy', psnr)
        epoch_end_time = time.time()
        np.save('./save_net/' + mname + '/G_loss.npy', G_loss)
        np.save('./save_net/' + mname + '/D_loss.npy', D_loss)
        per_epoch_ptime = epoch_end_time - epoch_start_time
        torch.save(G.state_dict(),
                   './save_net/' + mname + '/G_%d' % (epoch))
        torch.save(D.state_dict(),
                   './save_net/' + mname + '/D_%d' % (epoch))


##test
def test():
    ##data
    test_noisy_data = np.load('./data/' + test_name + '_noisy_img_var' + str(var) + '.npy')
    test_clean_data = np.load('./data/' + test_name + '_clean_img.npy')
    G = dcgan.generator(64).cuda()
    pretrained_net = torch.load(
        './save_net/' + mname + '/G_' + str(
            train_epoch - 1))  # torch.load('D:\cryo_em\code\denoise_code\denoise_gan\save_net\gan\model9\G_19')
    G.load_state_dict(pretrained_net)
    MSE = []
    PSNR = []
    img1 = []
    k = 0
    for i in range(int(600 / batch_size)):
        cond = test_noisy_data[i * batch_size: i * batch_size + batch_size, :, :]
        img = test_clean_data[i * batch_size: i * batch_size + batch_size, :, :]
        k = k + 1
        pcond2 = np.reshape(cond, (batch_size, 1, img_size, img_size))
        pcond2 = torch.from_numpy(pcond2)
        pcond2 = pcond2.to(device=device, dtype=dtype)
        for t in range(batch_size):
            gen_img = G(pcond2)[t].cpu().detach().numpy()
            gen_img = gen_img.reshape((img_size, img_size))
            m = criterion.MSE(gen_img, img[t])
            MSE.append(m)
            PSNR.append(10 * math.log10(1 / m))
        if i % 20 == 0:
            img1.append(gen_img)
            plt.imshow(gen_img, cmap='gray')
    np.save('./save_net/' + mname + '/test_mse.npy', MSE)
    np.save('./save_net/' + mname + '/test_psnr.npy', PSNR)


if __name__ == '__main__':
    train()
    #test()
