from collections import OrderedDict
import json
import string
import torch

import pyprob
from pyprob import Model
import pyprob.distributions as dists


def format_cc(cc, cc_format) -> str:
    if cc_format == 0: return ""
    elif cc_format == 1: return "+" + cc
    elif cc_format == 2: return "+" + cc
    elif cc_format == 3: return cc + "-"
    elif cc_format == 4: return "+" + cc + "-"
    elif cc_format == 5: return cc + " "
    else: return "+" + cc + " "

def format_ac(ac, ac_format) -> str:
    if ac_format == 0: return ac
    elif ac_format == 1: return "(" + ac + ")"
    elif ac_format == 2: return ac + "-"
    elif ac_format == 3: return "(" + ac + ")-"
    elif ac_format == 4: return ac + " "
    else: return "(" + ac + ") "

def format_line_number(line_number_blocks, line_format) -> str:
    if line_format == 0: return "".join(line_number_blocks)
    elif line_format == 1: return "-".join(line_number_blocks)
    else: return " ".join(line_number_blocks)

def letter_to_index(letter: str) -> int:
    index = ALL_LETTERS.find(letter)
    if index == -1: raise Exception(f"letter {letter} is not permitted.")
    return index

def pad_string(original: str, desired_len: int, pad_character: str = ' ') -> str:
    # Returns the padded version of the original string to length: desired_len
    return original + (pad_character * (desired_len - len(original)))

def load_json(jsonpath: str) -> dict:
    with open(jsonpath) as jsonfile:
        return json.load(jsonfile, object_pairs_hook=OrderedDict)


"""
Supports Countries In
- https://en.wikipedia.org/wiki/National_conventions_for_writing_telephone_numbers#International_Telecommunication_Union
"""
COUNTRY_INFO = load_json("./data/limited_cc.json")
MAX_STRING_LEN = 30
ALL_DIGITS = string.digits
ALL_LETTERS = string.digits+' .,:()+-'
N_DIGIT = len(ALL_DIGITS)
N_LETTER = len(ALL_LETTERS)


class OneHot2DCategorical(dists.Categorical):
    def sample(self):
        s = self._torch_dist.sample()
        one_hot = self._probs * 0
        for i, val in enumerate(s):
            one_hot[i, int(val.item())] = 1
        return one_hot
    
    def log_prob(self, x, *args, **kwargs):
        # vector of one hot vectors
        non_one_hot = torch.tensor([row.nonzero() for row in x])
        return super().log_prob(non_one_hot, *args, **kwargs)


class PhoneParser(Model):
    def __init__(self):
        super().__init__(name="Phone number with Unknown Format")

    def forward(self):
        country_index = int(pyprob.sample(dists.Categorical(torch.tensor([1/len(COUNTRY_INFO)]*len(COUNTRY_INFO)))).item())
        country_info = COUNTRY_INFO[country_index]

        # Obtain formatted country code
        country_code = country_info['cc']
        cc_format = int(pyprob.sample(dists.Categorical(torch.tensor([1/3] + [1/9]*6))).item())
        full_cc = format_cc(country_code, cc_format)
        
        structure_index = int(pyprob.sample(dists.Categorical(torch.tensor([1/len(country_info['structure'])]*len(country_info['structure'])))).item())
        number_structure = country_info['structure'][structure_index]

        # Obtain formatted area code
        area_code_len = number_structure[0]
        area_code = ""
        for _ in range(area_code_len):
            curr_digit = int(pyprob.sample(dists.Categorical(torch.tensor([1/N_DIGIT]*N_DIGIT))).item())
            area_code += str(curr_digit)
        ac_format = int(pyprob.sample(dists.Categorical(torch.tensor([1/6]*6))).item())
        full_ac = format_ac(area_code, ac_format)

        # Obtain formatted line number
        line_number_structure = number_structure[1:]
        line_number_block_len = len(line_number_structure)
        line_number_blocks = []
        for i in range(line_number_block_len):
            number_block_len = line_number_structure[i]
            number_block_digits = ""
            for _ in range(number_block_len):
                number = int(pyprob.sample(dists.Categorical(torch.tensor([1/N_DIGIT]*N_DIGIT))).item())
                number_block_digits += str(number)
            line_number_blocks.append(number_block_digits)
        line_number = " ".join(line_number_blocks)
        line_format = int(pyprob.sample(dists.Categorical(torch.tensor([1/3]*3))).item())
        full_line = format_line_number(line_number_blocks, line_format)

        # make a categorical distribution that observes each letter independently (like 30 independent categoricals)
        output = pad_string(original=full_cc+full_ac+full_line, desired_len=MAX_STRING_LEN)
        probs = torch.ones(MAX_STRING_LEN, N_LETTER)*0.001
        for i, letter in enumerate(output):
            probs[i, letter_to_index(letter)] = 1.
        pyprob.observe(OneHot2DCategorical(probs), name=f"phone_string")

        return output, {'country': country_info['country'],'country code': country_code, 'area code': area_code, 'line number': line_number}
    
    def get_observes(self, phone_string):
        if len(phone_string) > 30: raise Exception("Phone number string length cannot exceed 30.")
        one_hot = torch.zeros(MAX_STRING_LEN, N_LETTER)
        phone_string = pad_string(original=phone_string, desired_len=MAX_STRING_LEN)
        for i, letter in enumerate(phone_string):
            one_hot[i, letter_to_index(letter)] = 1.
        
        return {'phone_string': one_hot}
