import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys
import yaml

from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torchsummary import summary

from model import Encoder, ResidualBlock, Discriminator, Generator, weights_init_normal, LambdaLR
from datasets import ImageDataset
from torch.utils.tensorboard import SummaryWriter



import torch.nn as nn
import torch.nn.functional as F
import torch

torch.cuda.empty_cache()

def dump_config(config, path):
    file_path = os.path.join(path, "config_dump.yml")
    with open(file_path, "w") as config_dump:
        yaml.dump(config, config_dump, yaml.Dumper)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--decay_epoch", type=int, default=80, help="epoch from which to start lr decay")
parser.add_argument("--data_dir", type=str, default="/selfdriving_with_sim2real/data/", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=3, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=64, help="size of image height")
parser.add_argument("--img_width", type=int, default=64, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=20, help="interval between saving model checkpoints")
parser.add_argument("--n_downsample", type=int, default=5, help="number downsampling layers in encoder")
parser.add_argument("--n_residual", type=int, default=2, help="number of redidual blocks in encoder")
parser.add_argument("--saved_model_path", type=str, default="../artifacts/", help="Path from where you want to reload a model.")
parser.add_argument("--dim", type=int, default=32, help="number of filters in first encoder layer")
parser.add_argument("--exp_name", type=str, default="dt_unit", help="experiment name")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
date_and_time = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
artifacts_path = "../artifacts/"+date_and_time
log_path = artifacts_path + "/_tb/"
# Create sample and checkpoint directories
os.makedirs(artifacts_path + "/%s" % opt.exp_name, exist_ok=True)
os.makedirs(artifacts_path + "/saved_models/", exist_ok=True)
os.makedirs(artifacts_path + "/_tb/", exist_ok=True)
#Logs
writer = SummaryWriter(log_dir=log_path)


# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_pixel = torch.nn.L1Loss()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Dimensionality (channel-wise) of image embedding
shared_dim = opt.dim * 2 ** opt.n_downsample
#shared_dim = 32
# Initialize generator and discriminator
shared_E = ResidualBlock(features=shared_dim)
E1 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, residual=opt.n_residual, shared_block=shared_E)
E2 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, residual=opt.n_residual,shared_block=shared_E)
shared_G = ResidualBlock(features=shared_dim)
G1 = Generator(dim=shared_dim, n_upsample=opt.n_downsample, residual=opt.n_residual, shared_block=shared_G)
G2 = Generator(dim=shared_dim, n_upsample=opt.n_downsample, residual=opt.n_residual, shared_block=shared_G)
D1 = Discriminator(input_shape, n_downsample=opt.n_downsample)
D2 = Discriminator(input_shape, n_downsample=opt.n_downsample)

if cuda:
    E1 = E1.cuda()
    E2 = E2.cuda()
    G1 = G1.cuda()
    G2 = G2.cuda()
    D1 = D1.cuda()
    D2 = D2.cuda()
    criterion_GAN.cuda()
    criterion_pixel.cuda()

if opt.epoch != 0:
    # Load pretrained models
    E1.load_state_dict(torch.load("%/saved_models/%s/E1_%d.pth" % (opt, opt.exp_name, opt.epoch)))
    E2.load_state_dict(torch.load("%/saved_models/%s/E2_%d.pth" % (opt.exp_name, opt.epoch)))
    G1.load_state_dict(torch.load("%/saved_models/%s/G1_%d.pth" % (opt.exp_name, opt.epoch)))
    G2.load_state_dict(torch.load("%/saved_models/%s/G2_%d.pth" % (opt.exp_name, opt.epoch)))
    D1.load_state_dict(torch.load("%/saved_models/%s/D1_%d.pth" % (opt.exp_name, opt.epoch)))
    D2.load_state_dict(torch.load("%/saved_models/%s/D2_%d.pth" % (opt.exp_name, opt.epoch)))
else:
    # Initialize weights
    E1.apply(weights_init_normal)
    E2.apply(weights_init_normal)
    G1.apply(weights_init_normal)
    G2.apply(weights_init_normal)
    D1.apply(weights_init_normal)
    D2.apply(weights_init_normal)

# Loss weights
# lambda_0 = 10  # GAN
# lambda_1 = 2  # KL (encoded images)
# lambda_2 = 100  # ID pixel-wise
# lambda_3 = 0.1  # KL (encoded translated images)
# lambda_4 = 100  # Cycle pixel-wise
lambda_0 = 1  # GAN
lambda_1 = 0.01 # KL (encoded images)
lambda_2 = 10  # ID pixel-wise
lambda_3 = 0.01 # KL (encoded translated images)
lambda_4 = 10  # Cycle pixel-wise

with open( artifacts_path + "/saved_models/configs.txt", 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f
    for args in vars(opt):
        print(args, getattr(opt, args))

    gan_height = opt.img_height // 2 ** opt.n_downsample
    gan_width = opt.img_width // 2 ** opt.n_downsample
    print()
    print("lambda_0 # GAN = ", lambda_0)
    print("lambda_1 # KL (encoded images) = ", lambda_1)
    print("lambda_2 # ID pixel-wise = ", lambda_2 )
    print("lambda_3 # KL (encoded translated images) = ", lambda_3)
    print("lambda_4 # Cycle pixel-wise = ", lambda_4)
    print()

    print("------------------------shared E----------------------------")
    #summary(shared_E, (shared_dim))
    print("------------------------   E1   ----------------------------")
    summary(E1, input_shape)
    print("------------------------   E2   ----------------------------")
    summary(E2, input_shape)
    print("------------------------shared G----------------------------")
    #summary(shared_G, (shared_dim))
    print("------------------------   G1   ----------------------------")
    summary(G1, (shared_dim, gan_height, gan_width))
    print("------------------------   G2   ----------------------------")
    summary(G2, (shared_dim, gan_height, gan_width))
    print("------------------------   D1   ----------------------------")
    summary(D1, input_shape)
    print("------------------------   D2   ----------------------------")
    summary(D2, input_shape)
    sys.stdout = original_stdout
# Code backup
os.system('cp  train.py {}/'.format(artifacts_path + "/saved_models/"))
os.system('cp  model.py {}/'.format(artifacts_path + "/saved_models/"))

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(E1.parameters(), E2.parameters(), G1.parameters(), G2.parameters()),
    lr=opt.lr,
    betas=(opt.b1, opt.b2),
)
optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Image transformations
transforms_ = [
    transforms.Resize([opt.img_height, opt.img_width]),
    #transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Training data loader
dataloader = DataLoader(
    ImageDataset(opt.data_dir, transforms_=transforms_, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
# Test data loader
val_dataloader = DataLoader(
    ImageDataset(opt.data_dir, transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=5,
    shuffle=True,
    num_workers=1,
)


def sample_images(batches_done, epochs):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    X1 = Variable(imgs["sim"].type(Tensor))
    X2 = Variable(imgs["real"].type(Tensor))
    _, Z1 = E1(X1)
    _, Z2 = E2(X2)
    fake_X1 = G1(Z2)
    fake_X2 = G2(Z1)
    img_sample = torch.cat((X1.data, fake_X2.data, X2.data, fake_X1.data), 0)
    # save_image(img_sample, "images/%s/%s.png" % (opt.exp_name, batches_done), nrow=5, normalize=True)
    save_image(img_sample, artifacts_path + "/%s/%s_epoch%s.png" % (opt.exp_name, batches_done, epochs), nrow=5, normalize=True)

def log_sample_images(batches_done, epochs):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    X1 = Variable(imgs["sim"].type(Tensor))
    X2 = Variable(imgs["real"].type(Tensor))
    _, Z1 = E1(X1)
    _, Z2 = E2(X2)
    fake_X1 = G1(Z2)
    fake_same_X1 = G2(Z2)

    fake_X2 = G2(Z1)
    fake_same_X2 = G1(Z1)
    #----------------------
    _, inv_Z1 = E2(X1)
    _, inv_Z2 = E1(X2)
    inv_fake_X1 = G1(Z2)
    inv_fake_same_X1 = G2(Z2)

    inv_fake_X2 = G2(Z1)
    inv_fake_same_X2 = G1(Z1)


    img_sample = torch.cat((X1.data, fake_X2.data,fake_same_X2.data, X2.data, fake_X1.data, fake_same_X1.data), 0)
    img_sample = make_grid(img_sample, nrow=5, normalize= True)
    inv_img_sample = torch.cat((X1.data, inv_fake_X2.data, inv_fake_same_X2.data, X2.data, inv_fake_X1.data, inv_fake_same_X1.data), 0)
    inv_img_sample = make_grid(inv_img_sample, nrow=5, normalize= True)
    # save_image(img_sample, "images/%s/%s.png" % (opt.exp_name, batches_done), nrow=5, normalize=True)
    #save_image(img_sample, artifacts_path + "/%s/%s_epoch%s.png" % (opt.exp_name, batches_done, epochs), nrow=5, normalize=True)
    writer.add_image("Image/epoch_%".format(epochs),img_sample, epoch)
    writer.add_image("Image/epoch_inv_%".format(epochs), inv_img_sample, epoch)


def compute_kl(mu):
    mu_2 = torch.pow(mu, 2)
    loss = torch.mean(mu_2)
    return loss


# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        X1 = Variable(batch["sim"].type(Tensor))
        X2 = Variable(batch["real"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((X1.size(0), *D1.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((X1.size(0), *D1.output_shape))), requires_grad=False)
        # print("shape valid", *D1.output_shape)
        # print("shape fake", fake.shape)
        # -------------------------------
        #  Train Encoders and Generators
        # -------------------------------

        optimizer_G.zero_grad()

        # Get shared latent representation
        mu1, Z1 = E1(X1)
        mu2, Z2 = E2(X2)

        # Reconstruct images
        recon_X1 = G1(Z1)
        recon_X2 = G2(Z2)

        # Translate images
        fake_X1 = G1(Z2)
        fake_X2 = G2(Z1)

        # Cycle translation
        mu1_, Z1_ = E1(fake_X1)
        mu2_, Z2_ = E2(fake_X2)
        cycle_X1 = G1(Z2_)
        cycle_X2 = G2(Z1_)
        #
        # print("enc shape", Z1.shape)
        # print("fake1 shape", fake_X1.shape)
        # print("crit1 shape", D1(fake_X1).shape)
        # print("crit2 shape", D2(fake_X2).shape)

        # Losses
        loss_GAN_1 = lambda_0 * criterion_GAN(D1(fake_X1), valid)
        loss_GAN_2 = lambda_0 * criterion_GAN(D2(fake_X2), valid)
        loss_KL_1 = lambda_1 * compute_kl(mu1)
        loss_KL_2 = lambda_1 * compute_kl(mu2)
        loss_ID_1 = lambda_2 * criterion_pixel(recon_X1, X1)
        loss_ID_2 = lambda_2 * criterion_pixel(recon_X2, X2)
        loss_KL_1_ = lambda_3 * compute_kl(mu1_)
        loss_KL_2_ = lambda_3 * compute_kl(mu2_)
        loss_cyc_1 = lambda_4 * criterion_pixel(cycle_X1, X1)
        loss_cyc_2 = lambda_4 * criterion_pixel(cycle_X2, X2)

        # Total loss
        loss_G = (
            loss_KL_1
            + loss_KL_2
            + loss_ID_1
            + loss_ID_2
            + loss_GAN_1
            + loss_GAN_2
            + loss_KL_1_
            + loss_KL_2_
            + loss_cyc_1
            + loss_cyc_2
        )

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator 1
        # -----------------------

        optimizer_D1.zero_grad()

        loss_D1 = criterion_GAN(D1(X1), valid) + criterion_GAN(D1(fake_X1.detach()), fake)

        loss_D1.backward()
        optimizer_D1.step()

        # -----------------------
        #  Train Discriminator 2
        # -----------------------

        optimizer_D2.zero_grad()

        loss_D2 = criterion_GAN(D2(X2), valid) + criterion_GAN(D2(fake_X2.detach()), fake)

        loss_D2.backward()
        optimizer_D2.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
            % (epoch, opt.n_epochs, i, len(dataloader), (loss_D1 + loss_D2).item(), loss_G.item(), time_left)
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done, epoch)


    writer.add_scalar("Loss/loss_GAN_1",loss_GAN_1 ,epoch)
    writer.add_scalar("Loss/loss_GAN_2",loss_GAN_2 ,epoch)
    writer.add_scalar("Loss/loss_KL_1",loss_KL_1 ,epoch)
    writer.add_scalar("Loss/loss_KL_2",loss_KL_2 ,epoch)
    writer.add_scalar("Loss/loss_ID_1",loss_ID_1 ,epoch)
    writer.add_scalar("Loss/loss_ID_2",loss_ID_2 ,epoch)
    writer.add_scalar("Loss/loss_KL_1_",loss_KL_1_ ,epoch)
    writer.add_scalar("Loss/loss_KL_2_",loss_KL_2_ ,epoch)
    writer.add_scalar("Loss/loss_cyc_1",loss_cyc_1 ,epoch)
    writer.add_scalar("Loss/loss_cyc_2",loss_cyc_2 ,epoch)
    writer.add_scalar("Loss/loss_total",loss_G ,epoch)
    log_sample_images(batches_done, epoch)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D1.step()
    lr_scheduler_D2.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(E1.state_dict(), artifacts_path + "/saved_models/E1_%d.pth" % epoch)
        torch.save(E2.state_dict(), artifacts_path + "/saved_models/E2_%d.pth" % epoch)
        torch.save(G1.state_dict(), artifacts_path + "/saved_models/G1_%d.pth" % epoch)
        torch.save(G2.state_dict(), artifacts_path + "/saved_models/G2_%d.pth" % epoch)
        torch.save(D1.state_dict(), artifacts_path + "/saved_models/D1_%d.pth" % epoch)
        torch.save(D2.state_dict(), artifacts_path + "/saved_models/D2_%d.pth" % epoch)
