import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import initialize_weights

class DA(nn.Module):
    def __init__(self, patch_width, width_multiplier=2):
        super(DA, self).__init__()
        self.wd = width_multiplier
        self.pw = patch_width
        self.fc1 = nn.Linear(self.pw*self.pw, (self.wd*self.pw)**2)
        self.fc2 = nn.Linear((self.wd*self.pw)**2, self.pw**2)


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x


class ConvDa(nn.Module):
    def __init__(self, channel_multiplier=1):
        super(ConvDa, self).__init__()

        self.cm = channel_multiplier

        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 32*self.cm, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32*self.cm, 16*self.cm, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16*self.cm, 8*self.cm, 3, padding=1)
        self.relu3 = nn.ReLU()
#        self.pool3 = nn.MaxPool2d(2, 2)



        ## decoder layers ##
        # transpose layer, a kernel of 2 and a stride of 2 will increase the spatial dims by 2
#        self.t_conv1 = nn.ConvTranspose2d(8, 8, 2, stride=2)
#        self.relu4 = nn.ReLU()

        self.t_conv2 = nn.ConvTranspose2d(8*self.cm, 16*self.cm, 2, stride=2) # 8,16,2,2
        self.relu5 = nn.ReLU()

        self.t_conv3 = nn.ConvTranspose2d(16*self.cm, 32*self.cm, 2, stride=2)
        self.relu6 = nn.ReLU()

        self.conv_out = nn.Conv2d(32*self.cm, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # say x.shape = [1, 1, 32, 32]
        x = self.relu1(self.conv1(x))   # [1, 32, 32, 32]
        x = self.pool1(x)               # [1, 32, 16, 16]
        x = self.relu2(self.conv2(x))   # [1, 16, 16, 16]
        x = self.pool2(x)               # [1, 16, 8, 8]
        x = self.relu3(self.conv3(x))   # [1, 8, 8, 8]
#        x = self.pool3(x)               # [1, 8, 4, 4]

        ## decode ##
        # add transpose conv layers, with relu activation function
#        x = self.relu4(self.t_conv1(x)) # [1, 8, 8, 8]
        x = self.relu5(self.t_conv2(x)) # [1, 16, 16, 16]
        x = self.relu6(self.t_conv3(x)) # [1, 32, 32, 32]
        # transpose again, output should have a sigmoid applied
        x = self.sigmoid(self.conv_out(x))

        return x

class DVAE(nn.Module):

    def __init__(self, args):
        super(DVAE, self).__init__()
        self.pw = args.patch_width
        self.n_sample = args.n_sample
        self.reconstruction_function = nn.BCELoss()
        self.arch_type = args.arch_type
        self.z_dim = args.z_dim
        self.encoder_init()
        self.decoder_init()

    def loss_function(self, recon_x, x, mu, logsig):
        x_tile = x.repeat(self.n_sample,1,1,1,1).permute(1,0,2,3,4)
        J_low = self.elbo(recon_x, x_tile, mu, logsig)
        return J_low

    def elbo(self, recon_x, x, mu, logsig):
        N, M, C, w, h = x.shape
        x = x.contiguous().view([N * M, C, w, h])
        recon_x = recon_x.view([N * M, C, w, h])
        BCE = self.reconstruction_function(recon_x, x) / (N * M)
        KLD_element = (logsig - mu**2 - torch.exp(logsig) + 1 )
        KLD = - torch.mean(torch.sum(KLD_element * 0.5, dim=2))
        return BCE + KLD

    def encoder_init(self):
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        self.input_height = self.pw
        self.input_width = self.pw
        self.input_dim = 1

        if self.arch_type == 'conv':
            self.enc_layer1 = nn.Sequential(
                nn.Conv2d(self.input_dim, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
            )
            self.mu_fc = nn.Sequential(
                nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), self.z_dim),
            )

            self.sigma_fc = nn.Sequential(
                nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), self.z_dim),
            )
        else:

            self.enc_layer1 = nn.Sequential(
                nn.Linear(self.input_height * self.input_width, self.z_dim * 4),
                nn.BatchNorm1d(self.z_dim * 4),
                nn.LeakyReLU(0.2),
                nn.Linear(self.z_dim * 4, self.z_dim * 4),
                nn.BatchNorm1d(self.z_dim * 4),
                nn.LeakyReLU(0.2),
            )

            self.mu_fc = nn.Sequential(
                nn.Linear(self.z_dim * 4, self.z_dim),
            )

            self.sigma_fc = nn.Sequential(
                nn.Linear(self.z_dim * 4, self.z_dim),
            )

        initialize_weights(self)

    def encode(self, x):

        if self.arch_type == 'conv':
            x = self.enc_layer1(x)
            x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        else:
            x = x.view([-1, self.input_height * self.input_width * self.input_dim])
            x = self.enc_layer1(x)
        mean = self.mu_fc(x)
        sigma = self.sigma_fc(x)

        return mean, sigma

    def sample(self, mu, logsig):
        #std = logsig.mul(0.5).exp_()
        std = torch.exp(logsig*0.5)
        eps = torch.randn(std.size()).cuda()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decoder_init(self):
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        self.input_height = self.pw
        self.input_width = self.pw
        self.output_dim = 1

        if self.arch_type == 'conv':
            self.dec_layer1 = nn.Sequential(
                nn.Linear(self.z_dim, 128 * (self.input_height // 4) * (self.input_width // 4)),
                nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
                nn.ReLU(),
            )

            self.dec_layer2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
                nn.Sigmoid(),
            )
        else:

            self.dec_layer1 = nn.Sequential(
                nn.Linear(self.z_dim, self.z_dim * 4),
                nn.BatchNorm1d(self.z_dim * 4),
                nn.LeakyReLU(0.2),
                nn.Linear(self.z_dim * 4, self.z_dim * 4),
                nn.BatchNorm1d(self.z_dim * 4),
                # nn.LeakyReLU(0.2),
                nn.Tanh(),
            )

            self.dec_layer2 = nn.Sequential(
                nn.Linear(self.z_dim * 4, self.input_height * self.input_width),
                nn.Sigmoid(),
            )
        initialize_weights(self)

    def decode(self, z):

        N, T, D = z.size()
        x = self.dec_layer1(z.view([-1, D]))

        if self.arch_type == 'conv':
            x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
            x = self.dec_layer2(x)
        else:
            x = self.dec_layer2(x)
            x = x.view(-1, 1, self.input_height, self.input_width)
        return x.view([N, T, -1, self.input_width, self.input_height])

    def forward(self, x):
        mu, logsig = self.encode(x)
        mu = mu.repeat(self.n_sample, 1, 1).permute(1, 0, 2)
        logsig = logsig.repeat(self.n_sample, 1, 1).permute(1, 0, 2)
        z = self.sample(mu, logsig)
        res = self.decode(z)
        return res, mu, logsig, z

class Generator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(Generator, self).__init__()
        self.conv11 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv11_bn = nn.BatchNorm2d(d)
        self.conv12 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv12_bn = nn.BatchNorm2d(d * 2)
        self.conv13 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv13_bn = nn.BatchNorm2d(d * 4)
        self.conv14 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv14_bn = nn.BatchNorm2d(d * 8)
        self.conv15 = nn.Conv2d(d * 8, d * 16, 4, 2, 1)
        self.conv15_bn = nn.BatchNorm2d(d * 16)
        self.deconv0 = nn.ConvTranspose2d(d * 16, d * 8, 4, 2, 1)
        self.deconv0_bn = nn.BatchNorm2d(d * 8)
        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(d * 4)
        self.deconv2 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 2)
        self.deconv3 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)
        # self.deconv4_bn = nn.BatchNorm2d(1)


    # forward method
    def forward(self, input1):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.conv11_bn(self.conv11(input1)))
        x = F.relu(self.conv12_bn(self.conv12(x)))
        x = F.relu(self.conv13_bn(self.conv13(x)))
        x = F.relu(self.conv14_bn(self.conv14(x)))
        #x = F.relu(self.conv15_bn(self.conv15(x)))
        #x = F.relu(self.deconv0_bn(self.deconv0(x)))
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))

        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

'''
class VAE(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(VAE, self).__init__()
        self.conv11 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv11_bn = nn.BatchNorm2d(d)
        self.conv12 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv12_bn = nn.BatchNorm2d(d * 2)
        self.conv13 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv13_bn = nn.BatchNorm2d(d * 4)
        self.conv141 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv141_bn = nn.BatchNorm2d(d * 8)
        self.conv142 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv142_bn = nn.BatchNorm2d(d * 8)


        self.deconv0 = nn.ConvTranspose2d(d * 16, d * 8, 4, 2, 1)
        self.deconv0_bn = nn.BatchNorm2d(d * 8)
        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(d * 4)
        self.deconv2 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 2)
        self.deconv3 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)
        # self.deconv4_bn = nn.BatchNorm2d(1)


    def encode(self, x):
        x = F.relu(self.conv11_bn(self.conv11(x)))
        x = F.relu(self.conv12_bn(self.conv12(x)))
        x = F.relu(self.conv13_bn(self.conv13(x)))
        return F.relu(self.conv141_bn(self.conv141(x))), F.relu(self.conv142_bn(self.conv142(x)))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = F.relu(self.deconv1_bn(self.deconv1(z)))
        z = F.relu(self.deconv2_bn(self.deconv2(z)))
        z = F.relu(self.deconv3_bn(self.deconv3(z)))
        z = F.sigmoid(self.deconv4(z))
        return z

    # forward method
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

'''

'''
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()

        self.pw = args.patch_width
        self.z_dim = args.z_dim

        self.fc1 = nn.Linear(self.pw * self.pw , 800)
        self.fc21 = nn.Linear(800, self.z_dim)
        self.fc22 = nn.Linear(800, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, 800)
        self.fc4 = nn.Linear(800, self.pw * self.pw)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = torch.sigmoid(self.fc4(h3))
        return h4.view((h4.shape[0], 1, self.pw, self.pw))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.pw * self.pw))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
'''