import torch
import torch.nn as nn
import torch.functional as F

import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
from pyro.optim import Adam

from util.convert import strings_to_tensor, tensor_to_string
from phone_vae_infcomp import PhoneVAE, index_encode, index_encode_to_one_hot_encode

import os

TEST_DATASET = ["+1 604 922 5941"]

svae = PhoneVAE(batch_size=1)
svae.load_checkpoint(filename="infcomp.pth.tar")

optimizer = Adam({'lr': 0.001})
csis = pyro.infer.CSIS(svae.model, svae.guide, optimizer, num_inference_samples=10)


print(f"TEST_DATA: {TEST_DATASET[0]}")
test_dataset = index_encode(TEST_DATASET)
posterior = csis.run(observations={'x': test_dataset})
marginal = pyro.infer.EmpiricalMarginal(posterior,sites=["ext"])

csis_samples = [marginal().detach() for _ in range(10)]
from phone_vae_infcomp import EXT
for sample in csis_samples:
    print(f"EXTENSION: {EXT[sample.item()]}")

