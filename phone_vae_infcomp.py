import os

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn

from data_loader.data_loader import load_json
from neural_net.mlp import HiddenLayerMLP
from neural_net.rnn import Encoder, Decoder
from util.convert import letter_to_index, index_to_digit, strings_to_tensor, strings_to_probs, N_DIGIT, N_LETTER, SOS_CHAR, EOS_CHAR, SOS_tensor, EOS_tensor, format_ext, format_prefix, format_number, tensor_to_string, experiment


COUNTRY_TO_EXT = load_json("./data/phone.json")
EXT_TO_COUNTRY = {v: k for k, v in COUNTRY_TO_EXT.items()}
EXT = list(COUNTRY_TO_EXT.values())
HIDDEN_SIZE = 32
MAX_STRING_LEN = 35


def index_encode(numbers: list, batch_size: int = 1, padding: str = ' '):
    encoded = torch.zeros(MAX_STRING_LEN, batch_size)
    for num_idx, number in enumerate(numbers):
        for char_idx in range(MAX_STRING_LEN):
            character = number[char_idx] if char_idx < len(number) else padding
            encoded[char_idx,num_idx] = letter_to_index(character)
    return encoded

def index_encode_to_one_hot_encode(index_encode: list):
    n, d = index_encode.shape
    encoded = torch.zeros(n,d,N_LETTER)
    for i in range(n):
        for j in range(d):
            encoded[i,j,int(index_encode[i,j])] = 1
    return encoded



def generate_string(rnn, address, address_type, length=15, hidden_layer=None):
    """
    Given a character RNN, generate a digit by sampling RNN generated distribution per timestep
    """
    name = ""
    next_char = '0' # TODO Is there better alternatives????????????????????????????????????????????????
    if hidden_layer is None: hidden_layer = rnn.init_hidden()
    for _ in range(length):
        lstm_input = strings_to_tensor([next_char],1,number_only=True)
        next_char_probs, hidden_layer = rnn(lstm_input, hidden_layer)
        next_char_index = pyro.sample(f"{address_type}_{address}", dist.Categorical(next_char_probs)).item()
        next_char = index_to_digit(next_char_index)
        name += next_char
        address += 1
    return name, hidden_layer, address

def step_rnn(rnn, address, address_type, length=15, hidden_layer=None, custom_input_size=None):
    """
    Hacky function for sampling from RNNs with input/output dimension that is nont N_DIGIT or N_LETTER
    """
    name = ""
    next_char = '0'
    if hidden_layer is None: hidden_layer = rnn.init_hidden()
    for _ in range(length):
        lstm_input = strings_to_tensor([next_char],1,custom_input_size=custom_input_size)
        next_char_probs, hidden_layer = rnn(lstm_input, hidden_layer)
        next_char_index = pyro.sample(f"{address_type}_{address}", dist.Categorical(next_char_probs)).item()
        next_char = str(next_char_index)
        name += next_char
        address += 1
    return name, hidden_layer, address



class PhoneVAE(nn.Module):
    def __init__(self, batch_size=1, hidden_size=HIDDEN_SIZE, use_cuda=False):
        super(PhoneVAE, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.encoder_rnn = Encoder(input_size=N_LETTER, hidden_size=self.hidden_size, batch_size=self.batch_size)
        self.prefix_rnn = Decoder(input_size=N_DIGIT, hidden_size=self.hidden_size, batch_size=self.batch_size, output_size=N_DIGIT)
        self.number_rnn = Decoder(input_size=N_DIGIT, hidden_size=self.hidden_size, batch_size=self.batch_size, output_size=N_DIGIT)

        flattened_hidden_size = 2
        for d in self.encoder_rnn.init_hidden()[0].shape:
            flattened_hidden_size *= d

        self.ext_exists_mlp = HiddenLayerMLP(input_size=flattened_hidden_size, output_size=2)
        self.ext_format_mlp = HiddenLayerMLP(input_size=flattened_hidden_size, output_size=6)
        self.ext_mlp = HiddenLayerMLP(input_size=flattened_hidden_size, output_size=len(EXT))
        self.prefix_format_mlp = HiddenLayerMLP(input_size=flattened_hidden_size, output_size=6)
        self.number_len_mlp = HiddenLayerMLP(input_size=flattened_hidden_size, output_size=3)
        self.number_format_mlp = HiddenLayerMLP(input_size=flattened_hidden_size, output_size=3)

        # NEW
        self.prefix_len_mlp = HiddenLayerMLP(input_size=flattened_hidden_size, output_size=3)
        self.number_part_len_rnn = Decoder(input_size=3, hidden_size=4, batch_size=self.batch_size, output_size=3)
        # NEW

        if use_cuda: 
            self.cuda()

    def guide(self, observations={}):
        """
        1 Run observed values through RNN to obtain the hidden state
        2 Pass the hidden state to these neural nets in a sequential order, then sample latents z
        2.1 MLP that generates probability for ext_exists - then sample z ~ Bernoulli(MLP(hidden))
        2.2 MLP that generates probabilities for ext_format - then sample z ~ Categorical(MLP(hidden))
        2.3 MLP that generates probabilities for ext - then sample z ~ Categorical(MLP(hidden))
        2.4 MLP that generates probabilities for prefix_format - then sample z ~ Categorical(MLP(hidden))
        2.5 MLP that generates probabilities for number_len - then sample z ~ Categorical(MLP(hidden))
        2.6 MLP that generates probabilities for number_format - then sample z ~ Categorical(MLP(hidden))
        """
        pyro.module("encoder", self.encoder_rnn)
        pyro.module("prefix", self.prefix_rnn)
        pyro.module("number_part_len_rnn", self.number_part_len_rnn)
        pyro.module("number", self.number_rnn)
        #pyro.module("ext_exists", self.ext_exists_mlp)
        pyro.module("ext_format", self.ext_format_mlp)
        pyro.module("ext", self.ext_mlp)
        pyro.module("prefix_format", self.prefix_format_mlp)
        pyro.module("prefix_format", self.prefix_len_mlp)
        pyro.module("number_len", self.number_len_mlp)
        pyro.module("number_format", self.number_format_mlp)
        
        x = observations['x']
        x = index_encode_to_one_hot_encode(x)

        encoder_hidden = self.encoder_rnn.init_hidden()
        for i in range(x.shape[0]):
            # RNN requires 3 dimensional inputs
            _, encoder_hidden = self.encoder_rnn(x[i].unsqueeze(0), encoder_hidden)
        
        flattened_encoder_hidden = torch.cat((encoder_hidden[0].flatten(),encoder_hidden[1].flatten())).unsqueeze(0)
        ext_exists_probs = self.ext_exists_mlp(flattened_encoder_hidden)
        
        #ext_exists = pyro.sample("ext_exists", dist.Categorical(ext_exists_probs)).item()
        #if ext_exists:
        ext_format_probs = self.ext_format_mlp(flattened_encoder_hidden)
        pyro.sample("ext_format", dist.Categorical(ext_format_probs)).item()
        ext_probs = self.ext_mlp(flattened_encoder_hidden)
        ext = EXT[pyro.sample("ext", dist.Categorical(ext_probs)).item()]
        
        
        prefix_format_probs = self.prefix_format_mlp(flattened_encoder_hidden)
        pyro.sample("prefix_format", dist.Categorical(prefix_format_probs)).item()
        prefix_len_probs = self.prefix_len_mlp(flattened_encoder_hidden)
        prefix_len = pyro.sample("prefix_len", dist.Categorical(prefix_len_probs)).item() + 2
        prefix, prefix_hidden, _ = generate_string(self.prefix_rnn,0,"prefix",length=prefix_len,hidden_layer=encoder_hidden)

        number_parts = []
        number_len_probs = self.number_len_mlp(flattened_encoder_hidden)
        number_len = pyro.sample("number_len", dist.Categorical(number_len_probs)).item() + 2
        number_format_probs = self.number_format_mlp(flattened_encoder_hidden)
        pyro.sample("number_format", dist.Categorical(number_format_probs)).item()

        hidden_layer = prefix_hidden
        number_part_len_str, _, _ = step_rnn(self.number_part_len_rnn,0,"number_part_len",length=number_len,custom_input_size=3) # Only lengths 2-5 is possible
        for i, number_part_len in enumerate(number_part_len_str):
            number_part_len = int(number_part_len) + 2
            number_part, hidden_layer, _ = generate_string(self.number_rnn,0,f"number_part_{i}",length=number_part_len,hidden_layer=hidden_layer)
            number_parts.append(number_part)
        
        canonical_number = f"+{ext}-" if ext != "" else ""
        canonical_number += f"({prefix})-"+'-'.join(number_parts)
        return {'canonical_number': canonical_number}

    def model(self, observations={'x': '1234'}):
        """
        Parsed Format
        - Extension
        - Prefix
        - Number
        1 Sample whether extension exists or not z ~ Bernoulli(0.5)
        1.1 Sample extension format z ~ Categorical([...]) for ["","+","-","+-"," ","+ "]
        1.2 Sample extension number from z ~ Categorical([...]) for valid extensions
        2 Sample prefix format from z ~ Categorical([...]) for ["","()","-","()-"," ","() "]
        3 Sample prefix from RNN
        4 Sample length of main number components from z ~ Categorical([...]) for [1,2,3,4]
        5 Sample separator between main number components from z ~ Categorical([...]) for ["","-"," "]
        6 Sample main number components repeatedly from RNN
        7 One hot encode the assembled number and observe them against real data
        """

        ext, full_ext = "", ""
        #ext_exists = pyro.sample("ext_exists", dist.Categorical(torch.tensor([1/2]*2))).item()
        #if ext_exists:
        ext_format = pyro.sample("ext_format", dist.Categorical(torch.tensor([1/6]*6))).item()
        ext = pyro.sample("ext", dist.Categorical(torch.tensor([1/len(EXT)]*len(EXT)))).item()
        full_ext = format_ext(EXT[ext], ext_format)
        
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
        probs = strings_to_probs([output], MAX_STRING_LEN)
        
        x = observations['x']
        if type(x) == str: x = index_encode([x])
        return pyro.sample("x", dist.Categorical(probs), obs=x)
        #return output

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
            raise(f"No model in path {folder}")
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
        