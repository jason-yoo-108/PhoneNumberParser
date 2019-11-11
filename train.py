import sys

import matplotlib.pyplot as plt
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from data_loader.data_loader import load_json
from util.convert import strings_to_tensor, SOS_CHAR
from phone_vae import PhoneVAE

"""
Use SVI to train model/guide

USAGE: python train.py <config file path>
"""
config = load_json(sys.argv[1])
ADAM_CONFIG = config['adam']
CUDA = config['cuda']
NUM_EPOCHS = config['num_epochs']
SESSION_NAME = config['session_name']

MAX_STRING_LEN = 35
RECORD_EVERY = 1
"""
TEST_STRINGS = [
    "+44 (0) 745 55 26 372",
    "+44-745-55-71-361",
    "0785-55-05-261",
    "07155528537",
    "+44-715-55-95-418",
    "+447155525116",
    "+13094435311",
    "6048337213",
    "(202)820-4141",
    "646-717-2202"
]
"""
TEST_STRINGS = [
    "+1 604 250 1363",
    "+1 604 922 5941",
    "+1 604 337 1000",
    "+1 604 250 9999",
    "+1 604 922 1414",
    "+1 604 337 2654",
    "+1 604 250 9573",
    "+1 604 922 2543",
    "+1 604 337 5068"
]

svae = PhoneVAE(batch_size=1)
optimizer = Adam(ADAM_CONFIG)
svi = SVI(svae.model, svae.guide, optimizer, loss=Trace_ELBO())


"""
Train the model
"""
train_elbo = []
for e in range(NUM_EPOCHS):
    epoch_loss = 0.
    for string in TEST_STRINGS:
        # Pad input string differently than observed string so program doesn't get rewarded by making string short
        one_hot_string = strings_to_tensor([string], MAX_STRING_LEN)
        if CUDA: one_hot_string.cuda()
        svi.step(one_hot_string)
        epoch_loss += svi.step(one_hot_string)
    if e % RECORD_EVERY == 0:
        avg_epoch_loss = epoch_loss/len(TEST_STRINGS)
        print(f"Epoch #{e} Average Loss: {avg_epoch_loss}")
        train_elbo.append(avg_epoch_loss)
        epoch_loss = 0



plt.plot(train_elbo)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss")
plt.savefig(f"result/{SESSION_NAME}.png")

svae.save_checkpoint(filename=f"{SESSION_NAME}.pth.tar")
