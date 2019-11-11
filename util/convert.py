import string
import unicodedata

import torch

SOS_CHAR = "*"
EOS_CHAR = "#"
# Decided to remove " .,:" since we assume they only occur in separators
ALL_DIGITS = string.digits
N_DIGIT = len(ALL_DIGITS)

ALL_LETTERS = string.digits+SOS_CHAR+EOS_CHAR+' .,:()+-'
N_LETTER = len(ALL_LETTERS)



def index_to_letter(index: int) -> str:
    return ALL_LETTERS[index]

def letter_to_index(letter: str) -> int:
    return ALL_LETTERS.find(letter)

def index_to_digit(index: int) -> str:
    return ALL_DIGITS[index]

def digit_to_index(digit: str) -> int:
    return ALL_DIGITS.find(digit)


def pad_string(original: str, desired_len: int, pad_character: str = EOS_CHAR):
    """
    Returns the padded version of the original string to length: desired_len
    """
    return original + (pad_character * (desired_len - len(original)))

def experiment(tensor: list):
    """
    Convert a 3D tensor into a 2D form
    """
    new_tensor = torch.zeros(tensor.shape[0],tensor.shape[1])
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            one_index = float('inf')
            for k in range(tensor.shape[2]):
                if tensor[i][j][k] == 1.:
                    one_index = k
                    break
            new_tensor[i][j] = one_index
    return new_tensor

def strings_to_tensor(strings: list, max_name_len: int, number_only: bool = False, pad_char: str = None, custom_input_size: int = None):
    """
    Turn a list of strings into a tensor of one-hot letter vectors
    of shape: <max_name_len x len(strings) x n_letters>

    All strings are padded with '0's such that they have the length: desired_len
    If number_only is true, one-hot letter vectors only account for digits
    """
    if number_only:
        to_index_func = digit_to_index
        inner_len = N_DIGIT
    else: 
        to_index_func = letter_to_index
        inner_len = N_LETTER
    
    if custom_input_size is not None:
        to_index_func = lambda x: int(x)
        inner_len = custom_input_size

    if pad_char is None:
        strings = list(map(lambda name: pad_string(name, max_name_len), strings))
    else:
        strings = list(map(lambda name: pad_string(name, max_name_len, pad_char), strings))
    tensor = torch.zeros(max_name_len, len(strings), inner_len)
    for i_s, s in enumerate(strings):
        for i_char, letter in enumerate(s):
            tensor[i_char][i_s][to_index_func(letter)] = 1
    return tensor

def strings_to_probs(strings: list, max_name_len: int, true_index_prob: float = 0.99):
    """
    Turn a list of strings probabilities over rows where the element of the index
    of character has probability of 0.99 and others 0.01/(size(n_letters)-1)
    of shape: <max_name_len x len(strings) x n_letters>

    All strings are padded with '0's such that they have the length: desired_len
    If number_only is true, one-hot letter vectors only account for digits
    """
    strings = list(map(lambda name: pad_string(name, max_name_len), strings))
    default_index_prob = (1.-true_index_prob)/N_LETTER
    tensor = torch.zeros(max_name_len, len(strings), N_LETTER) + default_index_prob
    for i_s, s in enumerate(strings):
        for i_char, letter in enumerate(s):
            tensor[i_char][i_s][letter_to_index(letter)] += true_index_prob
    return tensor

def tensor_to_string(tensor: list) -> list:
    """
    For debugging purpose - converts an one hot tensor to a list of strings
    """
    strings = []
    for i_s in range(tensor.shape[1]):
        curr_str = ""
        for char_index in range(tensor.shape[0]):
            one_index = -1
            for char_value_index in range(tensor.shape[2]):
                if tensor[char_index,i_s, char_value_index] == 1:
                    one_index = char_value_index
                    break
            curr_str += index_to_letter(one_index)
        strings.append(curr_str)
    return strings

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS
    )

def SOS_tensor():
    return strings_to_tensor([SOS_CHAR], 1, number_only=True)

def EOS_tensor():
    return strings_to_tensor([EOS_CHAR], 1, number_only=True)

def format_ext(ext, ext_format) -> str:
    if ext_format == 0: return ext
    elif ext_format == 1: return "+" + ext
    elif ext_format == 2: return ext + "-"
    elif ext_format == 3: return "+" + ext + "-"
    elif ext_format == 4: return ext + " "
    else: return "+" + ext + " "

def format_prefix(prefix, prefix_format) -> str:
    if prefix_format == 0: return prefix
    elif prefix_format == 1: return "(" + prefix + ")"
    elif prefix_format == 2: return prefix + "-"
    elif prefix_format == 3: return "(" + prefix + ")-"
    elif prefix_format == 4: return prefix + " "
    else: return "(" + prefix + ") "

def format_number(number_parts, number_format) -> str:
    if number_format == 0: return "".join(number_parts)
    elif number_format == 1: return "-".join(number_parts)
    else: return " ".join(number_parts)
