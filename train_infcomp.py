import torch
import torch.nn as nn
import torch.functional as F

import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
from pyro.optim import Adam

from util.convert import strings_to_tensor
from phone_infcomp import PhoneCSIS

import os

CONTINUE_TRAINING = True
N_STEPS = 100
phone_csis = PhoneCSIS()
optimizer = Adam({'lr': 0.0005})

if CONTINUE_TRAINING: phone_csis.load_checkpoint(filename="infcomp.pth.tar")

csis = pyro.infer.CSIS(phone_csis.model, phone_csis.guide, optimizer, num_inference_samples=10, training_batch_size=30)

losses = []
for step in range(N_STEPS):
    loss = csis.step()
    losses.append(loss)
    if step%5 == 0: print(f"step: {step} - loss: {loss}")

import matplotlib.pyplot as plt
plt.plot(losses)
plt.title("Infcomp Loss")
plt.xlabel("step")
plt.ylabel("loss")
plt.savefig(f"result/infcomp.png")


phone_csis.save_checkpoint(filename=f"infcomp.pth.tar")
