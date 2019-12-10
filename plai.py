import torch

import pyprob
from pyprob import Model
import pyprob.distributions as dists

from data_loader.data_loader import load_json
from util.convert import strings_to_probs, N_DIGIT, N_LETTER, format_ext, format_prefix, format_number, pad_string, letter_to_index

COUNTRY_TO_EXT = load_json("./data/phone.json")
EXT = list(COUNTRY_TO_EXT.values())
MAX_STRING_LEN = 35

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
        ext_format = int(pyprob.sample(dists.Categorical(torch.tensor([1/6]*6))).item())
        ext_index = int(pyprob.sample(dists.Categorical(torch.tensor([1/len(EXT)]*len(EXT)))).item())
        full_ext = format_ext(EXT[ext_index], ext_format)
        
        prefix_format = int(pyprob.sample(dists.Categorical(torch.tensor([1/6]*6))).item())
        prefix_len = int(pyprob.sample(dists.Categorical(torch.tensor([1/3]*3))).item() + 2) # From 2 to 5
        prefix_digits = ""
        for _ in range(prefix_len):
            curr_digit = int(pyprob.sample(dists.Categorical(torch.tensor([1/N_DIGIT]*N_DIGIT))).item())
            prefix_digits += str(curr_digit)
        full_prefix = format_prefix(prefix_digits,prefix_format)

        number_parts = []
        number_format = int(pyprob.sample(dists.Categorical(torch.tensor([1/3]*3))).item())
        number_len = int(pyprob.sample(dists.Categorical(torch.tensor([.65,.25,.1]))).item() + 2) # From 2 to 5
        for _ in range(number_len):
            number_part_len = int(pyprob.sample(dists.Categorical(torch.tensor([1/3]*3))).item() + 2) # From 2 to 5
            number_part_digits = ""
            for _ in range(number_part_len):
                number = int(pyprob.sample(dists.Categorical(torch.tensor([1/N_DIGIT]*N_DIGIT))).item())
                number_part_digits += str(number)
            number_parts.append(number_part_digits)
        full_number = format_number(number_parts, number_format)

        output = pad_string(original=full_ext+full_prefix+full_number, desired_len=MAX_STRING_LEN)
        print(output)
        # make a categorical distribution that observes each letter independently (like 35 independent categoricals)
        probs = torch.ones(MAX_STRING_LEN, N_LETTER)*0.001
        for i, letter in enumerate(output):
            probs[i, letter_to_index(letter)] = 1.
        pyprob.observe(OneHot2DCategorical(probs), name=f"phone_string")
    
    def get_observes(self, phone_string):
        one_hot = torch.zeros(MAX_STRING_LEN, N_LETTER)
        phone_string = pad_string(original=phone_string, desired_len=MAX_STRING_LEN)
        for i, letter in enumerate(phone_string):
            one_hot[i, letter_to_index(letter)] = 1.
        
        return {'phone_string': one_hot}
