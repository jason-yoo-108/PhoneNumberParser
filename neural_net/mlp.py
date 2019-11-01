import torch
import torch.nn as nn

class HiddenLayerMLP(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Converts RNN's hidden layers into a discrete probability distribution
        """
        super(HiddenLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input):
        """
        SHAPE REQUIREMENT
        - input: <batch_size x input_size>
        - output: <batch_size x output_size>
        """
        output = self.fc1(input)
        output = self.softmax(output)
        return output
