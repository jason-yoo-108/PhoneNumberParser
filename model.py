import pyro
import pyro.distributions as dist
import torch

from data_loader.data_loader import load_json
from neural_net.rnn import Encoder, Decoder
from util.convert import index_to_letter, letter_to_index, strings_to_tensor, N_LETTER, SOS_CHAR, EOS_CHAR, format_ext, format_prefix, format_number


COUNTRY_TO_EXT = load_json("./data/phone.json")
EXT_TO_COUNTRY = {v: k for k, v in COUNTRY_TO_EXT.items()}
EXT = list(COUNTRY_TO_EXT.values())
HIDDEN_SIZE = 64

"""
TODO: Find out if it is necessary to write sample statements for separators
GUIDE: Use Encoder and have Decoders for ALL the sample statements
- ext_exists
- ext_format
- ext
- prefix_format
- number_len
- number_format
- char ... (not these - these are observed)
"""

"""
Test Strings
+44 (0) 745 55 26 372
+44-745-55-71-361
0785-55-05-261
07155528537
+44-715-55-95-418
+447155525116
+13094435311
6048337213
(202)820-4141
646-717-2202
"""


def generate_string(rnn, address, max_length=15):
    """
    Given a character RNN, generate a name by sampling RNN generated distribution per timestep
    """
    name = ""
    next_char = SOS_CHAR
    hidden_layer = rnn.hidden_layer
    for _ in range(max_length):
        if next_char == EOS_CHAR: break
        lstm_input = strings_to_tensor([next_char],1)
        next_char_probs, hidden_layer = rnn(lstm_input, hidden_layer)
        next_char_index = pyro.sample(f"char-{address}", dist.Categorical(next_char_probs)).item()
        next_char = index_to_letter(next_char_index)
        if next_char != SOS_CHAR and next_char != EOS_CHAR:
            name += next_char
        address += 1
    return name, hidden_layer, address


"""
Concern: Observations are only made at indexes that store prefix and main number digits

GUIDE: Use Encoder and have Decoders for ALL the sample statements
- ext_exists
- ext_format
- ext
- prefix_format
- number_len
- number_format
"""
def guide(x):
    """
    1 Run observed values through RNN to obtain the hidden state
    2 Pass the hidden state to these RNNs in a sequential order, then sample latents z
      2.1 RNN that generates probability for ext_exists - then sample z ~ Bernoulli(RNN(hidden))
      2.2 RNN that generates probabilities for ext_format - then sample z ~ Categorical(RNN(hidden))
      2.3 RNN that generates probabilities for ext - then sample z ~ Categorical(RNN(hidden))
      2.4 RNN that generates probabilities for prefix_format - then sample z ~ Categorical(RNN(hidden))
      2.5 RNN that generates probabilities for number_len - then sample z ~ Categorical(RNN(hidden))
      2.6 RNN that generates probabilities for number_format - then sample z ~ Categorical(RNN(hidden))
    """
    addr = 0
    encoder_rnn = Encoder(input_size=N_LETTER, hidden_size=HIDDEN_SIZE, batch_size=1)
    encoder_hidden = encoder_rnn.hidden_layer
    for i in range(x.shape[0]):
        _, encoder_hidden = encoder_rnn(x[i], encoder_hidden)
    
    # Hack to reuse Decoder for Bernoulli
    ext_exists_rnn = Decoder(input_size=N_LETTER, hidden_size=HIDDEN_SIZE, batch_size=1, output_size=2, num_layers=1)
    ext_exists_probs, _ = ext_exists_rnn(strings_to_tensor([SOS_CHAR],1), encoder_hidden)
    ext_exists = pyro.sample("ext_exists", dist.Bernoulli(ext_exists_probs[0])).item()
    if ext_exists:
        ext_format_rnn = Decoder(input_size=N_LETTER, hidden_size=HIDDEN_SIZE, batch_size=1, output_size=6, num_layers=1)
        ext_format_probs, _ = ext_format_rnn(strings_to_tensor([SOS_CHAR],1), encoder_hidden)
        ext_format = pyro.sample("ext_format", dist.Categorical(ext_format_probs)).item()
        ext_rnn = Decoder(input_size=N_LETTER, hidden_size=HIDDEN_SIZE, batch_size=1, output_size=len(EXT), num_layers=1)
        ext_probs, _ = ext_rnn(strings_to_tensor([SOS_CHAR],1), encoder_hidden)
        ext = pyro.sample("ext", dist.Categorical(ext_probs)).item()
        addr += len(format_ext(str(ext), ext_format))
    
    prefix_format_rnn = Decoder(input_size=N_LETTER, hidden_size=HIDDEN_SIZE, batch_size=1, output_size=6, num_layers=1)
    prefix_format_probs, _ = prefix_format_rnn(strings_to_tensor([SOS_CHAR],1), encoder_hidden)
    prefix_format = pyro.sample("prefix_format", dist.Categorical(prefix_format_probs)).item()
    if prefix_format % 2 == 1: addr += 1
    prefix_rnn = Decoder(input_size=N_LETTER, hidden_size=HIDDEN_SIZE, batch_size=1, output_size=N_LETTER)
    _, prefix_hidden, addr = generate_string(prefix_rnn,addr,max_length=4)
    if prefix_format in [1,2,4]: addr += 1
    if prefix_format in [3,5]: addr += 2

    number_parts = []
    number_len_rnn = Decoder(input_size=N_LETTER, hidden_size=HIDDEN_SIZE, batch_size=1, output_size=4, num_layers=1)
    number_len_probs, _ = number_len_rnn(strings_to_tensor([SOS_CHAR],1), prefix_hidden)
    number_len = pyro.sample("number_len", dist.Categorical(torch.tensor(number_len_probs))).item() + 1
    number_format_rnn = Decoder(input_size=N_LETTER, hidden_size=HIDDEN_SIZE, batch_size=1, output_size=3, num_layers=1)
    number_format_probs, _ = number_format_rnn(strings_to_tensor([SOS_CHAR],1), encoder_hidden)
    number_format = pyro.sample("number_format", dist.Categorical(number_format_probs)).item()

    hidden_layer = prefix_hidden
    for _ in range(number_len):
        number_rnn = Decoder(input_size=N_LETTER, hidden_size=HIDDEN_SIZE, batch_size=1, output_size=N_LETTER, hidden_layer=hidden_layer)
        number_part, hidden_layer, addr = generate_string(number_rnn,addr,max_length=4)
        number_parts.append(number_part)
        if number_format in [1,2]: addr += 1



def model(x):
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
    TODO: 7 One hot encode the assembled number and observe them against real data
    """
    addr = 0
    ext, full_ext = "", ""
    ext_exists = pyro.sample("ext_exists", dist.Bernoulli(0.5)).item()
    if ext_exists:
        ext_format = pyro.sample("ext_format", dist.Categorical(torch.tensor([1/6]*6))).item()
        ext = pyro.sample("ext", dist.Categorical(torch.tensor([1/len(EXT)]*len(EXT)))).item()
        full_ext = format_ext(str(ext), ext_format)
        addr += len(full_ext)
    
    prefix_format = pyro.sample("prefix_format", dist.Categorical(torch.tensor([1/6]*6))).item()
    if prefix_format % 2 == 1: addr += 1
    prefix_rnn = Decoder(input_size=N_LETTER, hidden_size=HIDDEN_SIZE, batch_size=1, output_size=N_LETTER)
    prefix, hidden_layer, addr = generate_string(prefix_rnn,addr,max_length=4)
    full_prefix = format_prefix(prefix,prefix_format)
    if prefix_format in [1,2,4]: addr += 1
    if prefix_format in [3,5]: addr += 2

    number_parts = []
    number_len = pyro.sample("number_len", dist.Categorical(torch.tensor([.01,.65,.25,.09]))).item() + 1
    number_format = pyro.sample("number_format", dist.Categorical(torch.tensor([1/3]*3))).item()
    for _ in range(number_len):
        number_rnn = Decoder(input_size=N_LETTER, hidden_size=HIDDEN_SIZE, batch_size=1, output_size=N_LETTER, hidden_layer=hidden_layer)
        number_part, hidden_layer, addr = generate_string(number_rnn,addr,max_length=4)
        number_parts.append(number_part)
        if number_format in [1,2]: addr += 1
    number = "-".join(number_parts)
    full_number = format_number(number_parts, number_format)

    output = full_ext + full_prefix + full_number

    return {
        "ext": ext,
        "prefix": prefix,
        "number": number,
        "full_number": output
    }


print(model(None))
