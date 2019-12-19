import torch
import torch.nn as nn
import torch.functional as F

import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
from pyro.optim import Adam

from data_loader.data_loader import load_json
from util.convert import strings_to_tensor, letter_to_index
from infcomp_pyro import PhoneCSIS, MAX_STRING_LEN

import os
import sys


config = load_json(sys.argv[1])
SESSION_NAME = config['session_name']
NUM_INFERENCE_SAMPLES = config['num_inference_samples']
NUM_POSTERIOR_SAMPLES = config['num_posterior_samples']
HIDDEN_SIZE = config['hidden_size'] if 'hidden_size' in config else 16

TEST_DATASET = [
    "+1 (604) 250 1363",
    "1-778-855-5941"
]


phone_csis = PhoneCSIS(hidden_size=HIDDEN_SIZE)
phone_csis.load_checkpoint(filename=f"infcomp_{SESSION_NAME}.pth.tar")

csis = pyro.infer.CSIS(phone_csis.model, phone_csis.guide, Adam({'lr': 0.001}), num_inference_samples=NUM_INFERENCE_SAMPLES)

from phone_infcomp import EXT
for phone_number in TEST_DATASET:
    print("=============================")
    print(f"Test Phone Number: {phone_number}")
    test_dataset = strings_to_tensor([phone_number],max_string_len=MAX_STRING_LEN,index_function=letter_to_index)
    posterior = csis.run(observations={'x': test_dataset})
    
    # Draw samples from the posterior
    # print(marginal._categorical.probs.T)
    # marginal = pyro.infer.EmpiricalMarginal(posterior, sites=static_sample_sites)
    print(f"Posterior Samples")
    csis_samples = [posterior() for _ in range(NUM_POSTERIOR_SAMPLES)]
    for sample in csis_samples:
        rv_names = sample.stochastic_nodes

        ext_format = sample.nodes['ext_format']['value'].item()
        ext_index = sample.nodes['ext_index']['value'].item()
        prefix_format = sample.nodes['prefix_format']['value'].item()
        prefix_len = sample.nodes['prefix_len']['value'].item()
        number_format = sample.nodes['number_format']['value'].item()
        number_len = sample.nodes['number_len']['value'].item()

        prefix_keys = sorted(list(filter(lambda name: 'prefix' in name and 'len' not in name and 'format' not in name, rv_names)))
        number_part_len_keys = sorted(list(filter(lambda name: 'number_part_len' in name, rv_names)))
        number_part_keys = sorted(list(filter(lambda name: 'number_part' in name and 'len' not in name, rv_names)))

        ext = EXT[ext_index]
        prefix = ""
        for key in prefix_keys:
            prefix += str(sample.nodes[key]['value'].item())
        number_parts = []
        for i,len_key in enumerate(number_part_len_keys):
            number_part_len = sample.nodes[len_key]['value'].item()
            number_part = ""
            for key in number_part_keys:
                if f"_{i}_" in key: number_part += str(sample.nodes[key]['value'].item())
            number_parts.append(number_part)
        print(f"{ext} ({prefix}) {'-'.join(number_parts)}")
    print("=============================")
