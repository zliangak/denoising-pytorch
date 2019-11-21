import torch
import torch.nn as nn
import torch.nn.functional as F



USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(generator, self).__init__()
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

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

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


class discriminator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv1_bn = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1_bn(self.conv1(input)), 0.01)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.01)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.01)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.01)
        x = self.conv5(x)
        x = F.sigmoid(x)

        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()