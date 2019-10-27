import string
import unicodedata

import torch

SOS_CHAR = "*"
EOS_CHAR = "#"
# Decided to remove " .,:" since we assume they only occur in separators
ALL_LETTERS = string.digits+SOS_CHAR+EOS_CHAR
N_LETTER = len(ALL_LETTERS)


def index_to_letter(index):
    return ALL_LETTERS[index]

def letter_to_index(letter):
    return ALL_LETTERS.find(letter)

def pad_string(original: str, desired_len: int, pad_character: str = ' '):
    """
    Returns the padded version of the original string to length: desired_len
    """
    return original + (pad_character * (desired_len - len(original)))

def strings_to_tensor(strings: list, max_name_len: int):
    """
    Turn a list of strings into a tensor of one-hot letter vectors
    of shape: <max_name_len x len(strings) x n_letters>

    All names are padded with ' ' such that they have the length: desired_len
    """
    strings = list(map(lambda name: pad_string(name, max_name_len), strings))
    tensor = torch.zeros(max_name_len, len(strings), N_LETTER)
    for i_s, s in enumerate(strings):
        for i_char, letter in enumerate(s):
            tensor[i_char][i_s][letter_to_index(letter)] = 1
    return tensor

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS
    )

def SOS_tensor():
    return strings_to_tensor(["*"], 1)

def EOS_tensor():
    return strings_to_tensor(["#"], 1)

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