import os

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn

from data_loader.data_loader import load_json
from neural_net.mlp import HiddenLayerMLP
from neural_net.rnn import Encoder, Decoder
from util.convert import strings_to_probs, N_DIGIT, N_LETTER, format_ext, format_prefix, format_number, to_rnn_tensor


COUNTRY_TO_EXT = load_json("./data/phone.json")
EXT_TO_COUNTRY = {v: k for k, v in COUNTRY_TO_EXT.items()}
EXT = list(COUNTRY_TO_EXT.values())
MAX_STRING_LEN = 35


def run_decoder(rnn, address_prefix, step_length, hidden_layer):
    """
    Get the decoder to generate characters for 'step_length' iterations.
    We assume decoders will only be generating digits.
    """
    generated = ""
    address_suffix = 0
    batch_size = 1

    # First rnn_input is start of sentence character
    next_char_index = -1
    for _ in range(step_length):
        rnn_input = torch.zeros(1, batch_size, rnn.input_size)
        rnn_input[0,0,next_char_index] = 1.
        next_char_probs, hidden_layer = rnn(rnn_input, hidden_layer)
        sampled_index = pyro.sample(f"{address_prefix}_{address_suffix}", dist.Categorical(next_char_probs)).item()
        generated += str(sampled_index)
        next_char_index = torch.argmax(next_char_probs, dim=2).item() # Next Character is the one with highest prob, not sampled ??????????????
        address_suffix += 1
    return generated, hidden_layer



class PhoneCSIS(nn.Module):
    def __init__(self, hidden_size=16, use_cuda=False):
        super(PhoneCSIS, self).__init__()

        self.encoder_rnn = Encoder(input_size=N_LETTER, hidden_size=hidden_size)
        # +1 in Decoder's input_size is for the Start of Sentence token
        self.prefix_rnn = Decoder(input_size=N_DIGIT+1, hidden_size=hidden_size, output_size=N_DIGIT)
        self.number_rnn = Decoder(input_size=N_DIGIT+1, hidden_size=hidden_size, output_size=N_DIGIT)
        self.number_part_len_rnn = Decoder(input_size=3+1, hidden_size=hidden_size, output_size=3)

        flattened_hidden_size = 2
        for d in self.encoder_rnn.blank_hidden_layer()[0].shape:
            flattened_hidden_size *= d

        self.ext_exists_mlp = HiddenLayerMLP(input_size=flattened_hidden_size, output_size=2)
        self.ext_format_mlp = HiddenLayerMLP(input_size=flattened_hidden_size, output_size=6)
        self.ext_mlp = HiddenLayerMLP(input_size=flattened_hidden_size, output_size=len(EXT))
        self.prefix_len_mlp = HiddenLayerMLP(input_size=flattened_hidden_size, output_size=3)
        self.prefix_format_mlp = HiddenLayerMLP(input_size=flattened_hidden_size, output_size=6)
        self.number_len_mlp = HiddenLayerMLP(input_size=flattened_hidden_size, output_size=3)
        self.number_format_mlp = HiddenLayerMLP(input_size=flattened_hidden_size, output_size=3)

        if use_cuda: self.cuda()


    def guide(self, observations={}):
        """
        Observations: A dictionary with a single key/value pair of form <'x': phone number tensor>
        1 Run observed values through the Encoder to obtain the hidden state
        2 pass the Encoder's hidden states to Decoders and MLPs to sample latents
        """
        pyro.module("encoder", self.encoder_rnn)
        pyro.module("prefix", self.prefix_rnn)
        pyro.module("number_part_len_rnn", self.number_part_len_rnn)
        pyro.module("number", self.number_rnn)
        pyro.module("ext_format", self.ext_format_mlp)
        pyro.module("ext_index", self.ext_mlp)
        pyro.module("prefix_format", self.prefix_format_mlp)
        pyro.module("prefix_format", self.prefix_len_mlp)
        pyro.module("number_len", self.number_len_mlp)
        pyro.module("number_format", self.number_format_mlp)
    
        x = to_rnn_tensor(observations['x'])
        encoder_hidden = self.encoder_rnn.blank_hidden_layer()
        for i in range(x.shape[0]):
            # RNN requires 3 dimensional inputs
            _, encoder_hidden = self.encoder_rnn(x[i].unsqueeze(0), encoder_hidden)
        
        flattened_encoder_hidden = torch.cat((encoder_hidden[0].flatten(),encoder_hidden[1].flatten())).unsqueeze(0)
        ext_format_probs = self.ext_format_mlp(flattened_encoder_hidden)
        ext_probs = self.ext_mlp(flattened_encoder_hidden)
        pyro.sample("ext_format", dist.Categorical(ext_format_probs)).item()
        ext = EXT[pyro.sample("ext_index", dist.Categorical(ext_probs)).item()]
        
        prefix_format_probs = self.prefix_format_mlp(flattened_encoder_hidden)
        prefix_len_probs = self.prefix_len_mlp(flattened_encoder_hidden)
        pyro.sample("prefix_format", dist.Categorical(prefix_format_probs)).item()
        prefix_len = pyro.sample("prefix_len", dist.Categorical(prefix_len_probs)).item() + 2
        prefix, prefix_hidden = run_decoder(self.prefix_rnn,'prefix',step_length=prefix_len,hidden_layer=encoder_hidden)
        number_parts = []
        number_len_probs = self.number_len_mlp(flattened_encoder_hidden)
        number_len = pyro.sample("number_len", dist.Categorical(number_len_probs)).item() + 2
        number_format_probs = self.number_format_mlp(flattened_encoder_hidden)
        pyro.sample("number_format", dist.Categorical(number_format_probs)).item()

        number_part_len_str, _ = run_decoder(self.number_part_len_rnn,'number_part_len',step_length=number_len,hidden_layer=prefix_hidden) # Only lengths 2-5 is possible
        hidden_layer = prefix_hidden
        for i, number_part_len in enumerate(number_part_len_str):
            number_part_len = int(number_part_len) + 2
            number_part, hidden_layer = run_decoder(self.number_rnn,f"number_part_{i}",step_length=number_part_len,hidden_layer=hidden_layer)
            number_parts.append(number_part)
        
        canonical_number = "" if ext == "" else f"+{ext}-"
        canonical_number += f"({prefix})-"+'-'.join(number_parts)
        return {'canonical_number': canonical_number}
    

    def model(self, observations={'x': torch.zeros(1,1)}):
        """
        Parsed Format: Extension, Prefix, Number
        1 Sample extension format z ~ Categorical([...]) for ["","+","-","+-"," ","+ "]
        2 Sample extension number from z ~ Categorical([...]) for valid extensions
        3 Sample prefix format from z ~ Categorical([...]) for ["","()","-","()-"," ","() "]
        4 Sample prefix from RNN
        5 Sample length of main number components from z ~ Categorical([...]) for [1,2,3,4]
        6 Sample separator between main number components from z ~ Categorical([...]) for ["","-"," "]
        7 Sample main number components repeatedly from RNN
        8 One hot encode the assembled number and observe them against real data
        """
        ext_format = pyro.sample("ext_format", dist.Categorical(torch.tensor([1/6]*6))).item()
        ext_index = pyro.sample("ext_index", dist.Categorical(torch.tensor([1/len(EXT)]*len(EXT)))).item()
        full_ext = format_ext(EXT[ext_index], ext_format)
        
        prefix_format = pyro.sample("prefix_format", dist.Categorical(torch.tensor([1/6]*6))).item()
        prefix_len = pyro.sample("prefix_len", dist.Categorical(torch.tensor([1/3]*3))).item() + 2 # From 2 to 5
        prefix_digits = ""
        for i in range(prefix_len):
            curr_digit = pyro.sample(f"prefix_{i}", dist.Categorical(torch.tensor([1/N_DIGIT]*N_DIGIT))).item()
            prefix_digits += str(curr_digit)        
        full_prefix = format_prefix(prefix_digits,prefix_format)

        number_parts = []
        number_len = pyro.sample("number_len", dist.Categorical(torch.tensor([.65,.25,.1]))).item() + 2 # From 2 to 5
        number_format = pyro.sample("number_format", dist.Categorical(torch.tensor([1/3]*3))).item()
        for i in range(number_len):
            number_part_len = pyro.sample(f"number_part_len_{i}", dist.Categorical(torch.tensor([1/3]*3))).item() + 2 # From 2 to 5
            number_part_digits = ""
            for j in range(number_part_len):
                number = pyro.sample(f"number_part_{i}_{j}", dist.Categorical(torch.tensor([1/N_DIGIT]*N_DIGIT))).item()
                number_part_digits += str(number)
            number_parts.append(number_part_digits)
        full_number = format_number(number_parts, number_format)

        output = full_ext + full_prefix + full_number
        probs = strings_to_probs(strings=[output], max_string_len=MAX_STRING_LEN)
        pyro.sample("x", dist.Categorical(probs), obs=observations['x'])


    def save_checkpoint(self, folder="nn_model", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        save_content = {
            'encoder_rnn': self.encoder_rnn.state_dict(),
            'prefix_rnn': self.prefix_rnn.state_dict(),
            'number_rnn': self.number_rnn.state_dict(),
            'ext_exists_mlp': self.ext_exists_mlp.state_dict(),
            'ext_format_mlp': self.ext_format_mlp.state_dict(),
            'ext_mlp': self.ext_mlp.state_dict(),
            'prefix_format_mlp': self.prefix_format_mlp.state_dict(),
            'number_format_mlp': self.number_format_mlp.state_dict(),
            'number_len_mlp': self.number_len_mlp.state_dict(),

            'prefix_len_mlp': self.prefix_len_mlp.state_dict(),
            'number_part_len_rnn': self.number_part_len_rnn.state_dict()
        }
        torch.save(save_content, filepath)

    def load_checkpoint(self, folder="nn_model", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath): 
            raise Exception(f"No model in path {folder}")
        save_content = torch.load(filepath)
        self.encoder_rnn.load_state_dict(save_content['encoder_rnn'])
        self.prefix_rnn.load_state_dict(save_content['prefix_rnn'])
        self.number_rnn.load_state_dict(save_content['number_rnn'])
        self.ext_exists_mlp.load_state_dict(save_content['ext_exists_mlp'])
        self.ext_format_mlp.load_state_dict(save_content['ext_format_mlp'])
        self.ext_mlp.load_state_dict(save_content['ext_mlp'])
        self.prefix_format_mlp.load_state_dict(save_content['prefix_format_mlp'])
        self.number_format_mlp.load_state_dict(save_content['number_format_mlp'])
        self.number_len_mlp.load_state_dict(save_content['number_len_mlp'])

        self.prefix_len_mlp.load_state_dict(save_content['prefix_len_mlp'])
        self.number_part_len_rnn.load_state_dict(save_content['number_part_len_rnn'])

"""
from util.convert import strings_to_tensor, letter_to_index
csis = PhoneCSIS()
for _ in range(1):
    x = strings_to_tensor(['+1 (604) 922 5941'], MAX_STRING_LEN, letter_to_index)
    print(csis.guide(observations={'x': x}))
"""