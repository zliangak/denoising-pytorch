import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train')
parser.add_argument('--lrd', type=float, default=0.001, help='learning rate of discriminator')
parser.add_argument('--lrg', type=float, default=0.01, help='learning rate of generator')
parser.add_argument('--device', type=str, default='cuda', help='device assignment')
parser.add_argument('--arch', type=str, default='jsgan', help='network architecture')
parser.add_argument('--test_name', type=str, default='test', help='name of test')
parser.add_argument('--train_name', type=str, default='train', help='name of train')
parser.add_argument('--var', type=float, default=10, help='variance of noise equal (var/255)**2')
parser.add_argument('--train_size', type=int, default=10000, help='learning rate of discriminator')

opt = parser.parse_args()
var = opt.var
mname = opt.arch + 'var' + str(var)
G_loss = np.load('./save_net/' + mname + '/G_loss.npy')
D_loss = np.load('./save_net/' + mname + '/D_loss.npy')
x = np.arange(300)
plt.plot(x, G_loss[0:300], label='G loss')
plt.plot(x, D_loss, label='D loss')
plt.xlabel('time')
plt.title('training loss')
plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
plt.savefig('loss_graph.png')
