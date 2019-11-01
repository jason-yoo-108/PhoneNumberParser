import sys

from data_loader.data_loader import load_json
from util.convert import strings_to_tensor, SOS_CHAR
from phone_vae import PhoneVAE

"""
Predict using SVAE with trained weights

USAGE: python test.py <config file path>
"""
config = load_json(sys.argv[1])
MAX_STRING_LEN = 35
SESSION_NAME = config['session_name']
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

print(SESSION_NAME)
svae = PhoneVAE(batch_size=1)
svae.load_checkpoint(filename=f"{SESSION_NAME}.pth.tar")

for string in TEST_STRINGS:
    # Pad input string differently than observed string so program doesn't get rewarded by making string short
    one_hot_string = strings_to_tensor([string], MAX_STRING_LEN, pad_char=SOS_CHAR)
    print(f"Original: {string}, Canonical: {svae.guide(one_hot_string)['canonical_number']}")
