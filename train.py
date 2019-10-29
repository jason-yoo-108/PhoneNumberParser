from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from util.convert import strings_to_tensor
from phone_vae import PhoneVAE



"""
Use SVI to train model/guide
"""
MAX_STRING_LEN = 25
NUM_EPOCHS = 50
CUDA = False
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


"""
Train the model
"""
train_elbo = []
for e in range(NUM_EPOCHS):
    epoch_loss = 0.
    for string in TEST_STRINGS:
        print(string)
        one_hot_string = strings_to_tensor([string], MAX_STRING_LEN)
        if CUDA: one_hot_string.cuda()
        svi.step(one_hot_string)
        epoch_loss += svi.step(one_hot_string)
    if e % 5 == 0:
        avg_epoch_loss = epoch_loss/len(TEST_STRINGS)
        train_elbo.append(avg_epoch_loss)
        epoch_loss = 0


import matplotlib.pyplot as plt

plt.plot(train_elbo)
plt.show()
