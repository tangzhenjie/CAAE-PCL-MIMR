import os
import argparse
import torch
import time
import datetime
import itertools
import pickle as pk
import numpy as np

from model_definition import Encoder
from model_definition import Decoder
from model_definition import Discriminator

from plot_tools import plot_pred



def load_data(opt):
    with open('../dataset/train_K_data/kds.pkl', 'rb') as file:
      kds = np.expand_dims(np.asarray(pk.load(file)), axis=1)
    print('Total number of conductivity images:', len(kds))

    x_train = kds[:opt.n_train]
    x_test = kds[opt.n_train: opt.n_train + opt.n_test]

    print("total training data shape: {}".format(x_train.shape))

    data = torch.utils.data.TensorDataset(torch.FloatTensor(x_train))
    data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size,
                                              shuffle=True, num_workers=int(2))
    return data_loader, x_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    parser.add_argument("--current-dir", type=str, default="./", help="data directory")
    parser.add_argument("--n-epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument('--n-train', type=int, default=23000, help='number of training data')
    parser.add_argument('--n-test', type=int, default=4000, help='number of training data')
    parser.add_argument("--batch-size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--lw", type=float, default=0.01, help="adversarial loss weight")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--sample-interval", type=int, default=10, help="interval between image sampling")
    opt = parser.parse_args()
    print(opt)

    current_datetime = datetime.datetime.now()

    date = 'exp_' + current_datetime.strftime("%Y%m%d%H")  # %M%S
    exp_dir = opt.current_dir + date + "/N{}_Bts{}_Eps{}_lr{}_lw{}".\
        format(opt.n_train, opt.batch_size, opt.n_epochs, opt.lr, opt.lw)

    output_dir = exp_dir + "/predictions"
    model_dir = exp_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Hyperparameter settings
    torch_device = torch.device('cpu')
    if torch.cuda.is_available():
        torch_device = torch.device('cuda')
        print('Default GPU is ' + torch.cuda.get_device_name(torch.device('cuda')))
    print('Running on ' + str(torch_device))

    # loss functions
    adversarial_loss = torch.nn.BCELoss()
    pixelwise_loss = torch.nn.L1Loss()

    nf, d, h, w = 2, 2, 11, 21

    # Initialize generator and discriminator
    encoder = Encoder(outchannels=nf)
    decoder = Decoder(inchannels=nf)
    discriminator = Discriminator(inchannels=nf)

    print("number of parameters: {}".format(encoder._n_parameters() + decoder._n_parameters() + discriminator._n_parameters()))

    if str(torch_device) == "cuda":
        encoder.cuda()
        decoder.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        pixelwise_loss.cuda()

    dataloader, x_test = load_data(opt)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    start = time.time()
    # ----------
    #  Training
    # ----------
    for epoch in range(1, opt.n_epochs+1):
        encoder.train()
        decoder.train()
        discriminator.train()

        for i, (imgs,) in enumerate(dataloader):

            # Adversarial ground truths
            valid = torch.ones((imgs.shape[0], 1), device=torch_device, requires_grad=False)
            fake = torch.zeros((imgs.shape[0], 1), device=torch_device, requires_grad=False)

            # Configure input
            real_imgs = imgs.to(torch_device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            encoded_imgs, _, _ = encoder(real_imgs)
            decoded_imgs = decoder(encoded_imgs)

            # Loss measures generator's ability to fool the discriminator
            g_loss_a = adversarial_loss(discriminator(encoded_imgs), valid)
            g_loss_c = pixelwise_loss(decoded_imgs, real_imgs)

            g_loss = opt.lw * g_loss_a + g_loss_c

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Sample noise as discriminator ground truth
            z = torch.as_tensor(np.random.normal(0, 1, (imgs.shape[0], nf, d, h, w)), dtype=torch.float, device=torch_device)

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(z), valid)
            fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f /G_A loss: %f/ G_C loss: %f]"
            % (
            epoch, opt.n_epochs, i + 1, len(dataloader), d_loss.item(), g_loss.item(), g_loss_a.item(), g_loss_c.item())
        )

        if epoch % opt.sample_interval == 0:
            encoder.eval()
            decoder.eval()

            n_samples = 1
            z = torch.as_tensor(np.random.normal(0, 1, (n_samples, nf, d, h, w)), dtype=torch.float,
                                device=torch_device)
            gen_imgs = decoder(z)
            samples = np.squeeze(gen_imgs.data.cpu().numpy())
            plot_pred(samples, epoch, 0, output_dir)

            idx = np.random.choice(opt.n_test, 1, replace=False)
            real_imgs = x_test[idx]
            real_imgs = torch.as_tensor(real_imgs, dtype=torch.float, device=torch_device)
            encoded_imgs, _, _ = encoder(real_imgs)
            decoded_imgs = decoder(encoded_imgs)
            samples_gen = np.squeeze(decoded_imgs.data.cpu().numpy())
            samples_real = np.squeeze(real_imgs.data.cpu().numpy())

            samples = np.vstack((samples_real[:3], samples_gen[:3], samples_real[3:], samples_gen[3:]))
            plot_pred(samples, epoch, idx, output_dir)

    torch.save(decoder.state_dict(), model_dir + '/AAE_decoder_epoch{}.pth'.format(opt.n_epochs))
    torch.save(encoder.state_dict(), model_dir + '/AAE_encoder_epoch{}.pth'.format(opt.n_epochs))
    torch.save(discriminator.state_dict(), model_dir + '/AAE_discriminator_epoch{}.pth'.format(opt.n_epochs))
    print('time for training:', time.time()-start)