import torch
import torch.nn as nn
import torch.functional as F

import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
from pyro.optim import Adam

from util.convert import strings_to_tensor, letter_to_index
from phone_infcomp import PhoneCSIS, MAX_STRING_LEN

import os

NN_FILENAME = "infcomp_best.pth.tar"
NUM_SAMPLES = 100
TEST_DATASET = ["+1 (604) 922 5941"]

phone_csis = PhoneCSIS()
phone_csis.load_checkpoint(filename=NN_FILENAME)

csis = pyro.infer.CSIS(phone_csis.model, phone_csis.guide, Adam({'lr': 0.001}), num_inference_samples=NUM_SAMPLES)

from phone_infcomp import EXT

for phone_number in TEST_DATASET:
    print("=============================")
    print(f"Phone Number: {phone_number}")
    test_dataset = strings_to_tensor([phone_number],max_string_len=MAX_STRING_LEN,index_function=letter_to_index)
    posterior = csis.run(observations={'x': test_dataset})

    for _ in range(10):
        sample = posterior._sample_from_joint()
        rv_names = sample.stochastic_nodes
        ext = EXT[sample.nodes['ext_index']['value'].item()]
        prefix_len = sample.nodes['prefix_len']['value'].item()+2
        prefix = ""
        for i in range(prefix_len):
            prefix += str(sample.nodes[f'prefix_{i}']['value'].item())
        number_len = sample.nodes['number_len']['value'].item()+2
        number_parts = []
        for i in range(number_len):
            number_part = ""
            number_part_len = sample.nodes[f'number_part_len_{i}']['value'].item()+2
            for j in range(number_part_len):
                number_part += str(sample.nodes[f'number_part_{i}_{j}']['value'].item())
            number_parts.append(number_part)
        print(f"Canonical Number: +{ext} ({prefix}) {'-'.join(number_parts)}")
    print("=============================")

"""
* Old way for obtaining guide result
    marginal = pyro.infer.EmpiricalMarginal(posterior,sites=["ext_index"])
    print(marginal._categorical.probs.T)
    csis_samples = [marginal().detach() for _ in range(20)]
    for sample in csis_samples:
        print(f"Extension: {EXT[sample.item()]}")
"""