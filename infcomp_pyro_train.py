import torch
import torch.nn as nn
import torch.functional as F

import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
from pyro.optim import Adam

from data_loader.data_loader import load_json
from util.convert import strings_to_tensor
from infcomp_pyro import PhoneCSIS

import os
import sys

pyro.enable_validation(True)

"""
Usage: python train_infcomp.py <Config File Path>
"""
config = load_json(sys.argv[1])
ADAM_CONFIG = config['adam']
CUDA = config['cuda']
NUM_EPOCHS = config['num_epochs']
SESSION_NAME = config['session_name']
CONTINUE_TRAINING = config['continue_training']
HIDDEN_SIZE = config['hidden_size'] if 'hidden_size' in config else 16

phone_csis = PhoneCSIS(hidden_size=HIDDEN_SIZE)
optimizer = Adam(ADAM_CONFIG)
if CONTINUE_TRAINING: phone_csis.load_checkpoint(filename=f"infcomp_{SESSION_NAME}.pth.tar")

csis = pyro.infer.CSIS(phone_csis.model, phone_csis.guide, optimizer, num_inference_samples=10, training_batch_size=30)
losses = []
for step in range(NUM_EPOCHS):
    loss = csis.step()
    if step%5 == 0: 
        print(f"step: {step} - loss: {loss}")
        losses.append(loss)
    if step>0 and step%250 == 0:
        print(f"Saving plot to result/infcomp_{SESSION_NAME}.png...")
        plt.plot(losses)
        plt.title("Infcomp Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(f"result/infcomp_{SESSION_NAME}.png")
        phone_csis.save_checkpoint(filename=f"infcomp_{SESSION_NAME}.pth.tar")

print(f"Saving plot to result/infcomp_{SESSION_NAME}.png...")
plt.plot(losses)
plt.title("Infcomp Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(f"result/infcomp_{SESSION_NAME}.png")

phone_csis.save_checkpoint(filename=f"infcomp_{SESSION_NAME}.pth.tar")
