from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from util.convert import strings_to_tensor
from phone_vae import PhoneVAE


"""
Use SVI to train model/guide
"""
MAX_STRING_LEN = 21
NUM_EPOCHS = 50
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

svae = PhoneVAE(batch_size=len(TEST_STRINGS))
optimizer = Adam({"lr": 1.e-3})
svi = SVI(svae.model, svae.guide, optimizer, loss=Trace_ELBO())

def train(svi, test_strings, cuda=False):
    """
    Screw epoch and just run SVI on the whole dataset at once
    """
    epoch_loss = 0.
    one_hot_strings = strings_to_tensor(test_strings, MAX_STRING_LEN)
    if cuda: one_hot_strings.cuda()

    epoch_loss += svi.step(one_hot_strings)
    avg_loss = epoch_loss / len(test_strings)
    return avg_loss

train_elbo = []
for i in range(NUM_EPOCHS):
    avg_loss = train(svi, TEST_STRINGS)
    if i % 5 == 0:
        train_elbo.append(avg_loss)

import matplotlib.pyplot as plt

plt.plot(train_elbo)
plt.show()