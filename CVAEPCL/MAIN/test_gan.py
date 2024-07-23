import torch
import datetime
import time
import pickle as pk
import numpy as np

from model_definition import Encoder
from model_definition import Decoder

# Hyperparameter settings
torch_device = torch.device('cpu')
if torch.cuda.is_available():
    torch_device = torch.device('cuda')
    print('Default GPU is ' + torch.cuda.get_device_name(torch.device('cuda')))
print('Running on ' + str(torch_device))

n_train = 23000
batch_size = 64
n_epochs = 50
lr = 0.0002 ## adam learning rate
lw = 0.01 ## "adversarial loss weight"

current_dir = "./"
current_datetime = datetime.datetime.now()
date = 'exp_2024063011' #+ current_datetime.strftime("%Y%m%d")  # %H%M%S

exp_dir = current_dir + date + "/N{}_Bts{}_Eps{}_lr{}_lw{}".\
    format(n_train, batch_size, n_epochs, lr, lw)

output_dir = exp_dir + "/predictions"
model_dir = exp_dir

# loss functions
pixelwise_loss = torch.nn.L1Loss()

nf, d, h, w = 2, 2, 11, 21

# Initialize generator and discriminator
encoder = Encoder(outchannels=nf)
decoder = Decoder(inchannels=nf)

encoder.load_state_dict(torch.load(model_dir + '/AAE_encoder_epoch{}.pth'.format(n_epochs)))
decoder.load_state_dict(torch.load(model_dir + '/AAE_decoder_epoch{}.pth'.format(n_epochs)))

if str(torch_device) == "cuda":
    encoder.cuda()
    decoder.cuda()
    pixelwise_loss.cuda()

encoder.eval()
decoder.eval()


## load the testing images
with open('../dataset/test_K_data/test_kds.pkl', 'rb') as file:
    kds = np.expand_dims(np.asarray(pk.load(file)), axis=1)[:2200]

# test the l1 loss
start = time.time()
g_loss = 0
for i in range(len(kds)//50):
    # Configure input
    real_imgs = torch.as_tensor(kds[i*50:(i+1)*50], dtype=torch.float, device=torch_device)
    encoded_imgs, _, _ = encoder(real_imgs)
    decoded_imgs = decoder(encoded_imgs)

    g_loss_c = pixelwise_loss(decoded_imgs, real_imgs).item()
    del encoded_imgs, decoded_imgs, real_imgs
    torch.cuda.empty_cache()
    g_loss += g_loss_c

print('time for processing data:', time.time()-start)
print('testing L1 loss on 2200 images:', g_loss/(len(kds)//50))

