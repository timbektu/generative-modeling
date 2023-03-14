import argparse
import os
from utils import get_args

import torch.nn as nn
import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model

#TODO: why is there no min-max shit happening around here?

def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    """
    TODO 1.3.1: Implement GAN loss for discriminator.
    Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    """

    discrim_real = discrim_real.reshape(-1)
    loss_real = nn.BCEWithLogitsLoss()(discrim_real, torch.ones_like(discrim_real))

    discrim_fake = discrim_fake.reshape(-1)
    loss_fake = nn.BCEWithLogitsLoss()(discrim_fake, torch.zeros_like(discrim_fake))
    return (loss_real + loss_fake) #TODO: multiply by weight?


def compute_generator_loss(discrim_fake):
    """
    TODO 1.3.1: Implement GAN loss for generator.
    """
    discrim_fake = discrim_fake.reshape(-1)
    loss_fake = nn.BCEWithLogitsLoss()(discrim_fake, torch.ones_like(discrim_fake))
    return loss_fake


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.3.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
