import torch.nn as nn
import torch

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    eps = torch.randn_like(std)
    #randn_like: Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.
    #return: random gaussian sample from distribution with mu and exp(logvar/2)
    return mu + eps*std


class ResidualBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale

        self.block = nn.Sequential(
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True),
            nn.Conv3d(filters, filters, 3, 1, 1, bias=True),
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True),
            nn.Conv3d(filters, filters, 3, 1, 1, bias=True)
        )

    def forward(self, x):
        out = self.block(x)
        return out.mul(self.res_scale) + x


class ResidualInResidualBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.blocks = nn.Sequential(
            ResidualBlock(filters), ResidualBlock(filters), ResidualBlock(filters)
        )

    def forward(self, x):
        return self.blocks(x).mul(self.res_scale) + x

class Encoder(nn.Module):
    def __init__(self, inchannels=1, outchannels=2, filters=48, num_res_blocks=2):
        super(Encoder, self).__init__()
        # input size, inchannels x 6 x 41 x 81
        self.conv1 = nn.Conv3d(inchannels, filters, kernel_size=3, stride=2, padding=1)
        # state size. filters x 3 x 21 x 41
        self.res_blocks1 = nn.Sequential(*[ResidualInResidualBlock(filters) for _ in range(num_res_blocks)])
        # state size. filters x 3 x 21 x 41
        self.trans1 = nn.Sequential(
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True),
            nn.Conv3d(filters, filters, kernel_size=3, stride=2, padding=1),
        )
        # state size. filters x 2 x 11 x 21
        self.mu = nn.Conv3d(filters, outchannels, 3, 1, 1, bias=False) #does not change state size.
        self.logvar = nn.Conv3d(filters, outchannels, 3, 1, 1, bias=False) #does not change state size.

    def forward(self, img):
        # img: inchannels x 6 x 41 x 81
        out1 = self.conv1(img)        # filters x 3 x 21 x 41
        out2 = self.res_blocks1(out1)   # filters x 3 x 21 x 41
        out3 = self.trans1(out2)        # filters x 2 x 11 x 21

        mu, logvar = self.mu(out3), self.logvar(out3)
        z = reparameterization(mu, logvar)  # latent dimension: outchannels x 2 x 11 x 21

        return z, mu, logvar

    def _n_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            n_params += param.numel()
        return n_params

class Decoder(nn.Module):
    def __init__(self, inchannels=2, outchannels=1, filters=48, num_res_blocks=1):
        super(Decoder, self).__init__()

        # First layer. input size, inchannels x 2 x 11 x 21
        self.conv1 = nn.Conv3d(inchannels, filters, kernel_size=3, stride=1, padding=1)

        # state size. filters x 2 x 11 x 21
        # Residual blocks
        self.res_block1 = nn.Sequential(*[ResidualInResidualBlock(filters) for _ in range(num_res_blocks + 1)])
        self.transup1 = nn.Sequential(
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(4, 21, 41), mode='nearest'),
        )
        self.res_block2 = nn.Sequential(*[ResidualInResidualBlock(filters) for _ in range(num_res_blocks)])
        self.transup2 = nn.Sequential(
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(8, 41, 81), mode='nearest'),
            nn.Conv3d(filters, outchannels, kernel_size=3, stride=1, padding=(0, 1, 1))
        )

    def forward(self, z):
        # x: in_channels x 2 x 11 x 21
        out1 = self.conv1(z)          # filters x 2 x 11 x 21
        out2 = self.res_block1(out1)   # filters x 2 x 11 x 21
        out3 = self.transup1(out2)      # filters x 4 x 21 x 41
        out4 = self.res_block2(out3)   # filters x 4 x 21 x 41
        img = self.transup2(out4)     # filters x 6 x 41 x 81

        return img

    def _n_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            n_params += param.numel()
        return n_params

class Discriminator(nn.Module):
    def __init__(self, inchannels=2, outchannels=1, filters=48):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (inchannels) x 2 x 11 x 21
            nn.Conv3d(inchannels, filters, 3, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (filters) x 1 x 6 x 11
            nn.Conv3d(filters, filters, 3, 1, 1, bias=True),
            nn.BatchNorm3d(filters),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (filters) x 1 x 6 x 11
        )

        self.fc1 = nn.Sequential(
            nn.Linear(filters * 6 * 11,128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, outchannels),
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.main(input)
        output = output.view(output.size(0), -1)
        output1 = self.fc1(output)
        output2 = self.fc2(output1)
        return output2

    def _n_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            n_params += param.numel()
        return n_params