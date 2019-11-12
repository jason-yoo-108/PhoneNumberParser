import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Takes in an one-hot tensor of names and produces hidden state and cell state
    for decoder LSTM to use.

    input_size: N_LETTER
    hidden_size: Size of the hidden dimension
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Initialize LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, input, hidden):
        """
        Run LSTM through 1 time step.

        SHAPE REQUIREMENT
        - input: <1 x batch_size x N_LETTER>
        - hidden: (<num_layer x batch_size x hidden_size>, <num_layer x batch_size x hidden_size>)
        """
        lstm_out, hidden = self.lstm(input, hidden)
        return lstm_out, hidden
    
    def blank_hidden_layer(self, batch_size=1):
        return (torch.zeros(self.num_layers,batch_size,self.hidden_size),
                torch.zeros(self.num_layers,batch_size,self.hidden_size))



class Decoder(nn.Module):
    """
    Accept hidden layers as an argument <num_layer x batch_size x hidden_size> for each hidden and cell state.
    At every forward call, output probability vector of <batch_size x output_size>.

    input_size: N_LETTER
    hidden_size: Size of the hidden dimension
    output_size: N_LETTER
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Initialize LSTM - Notice it does not accept output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input, hidden):
        """
        Run LSTM through 1 time step

        SHAPE REQUIREMENT
        - input: <1 x batch_size x N_LETTER>
        - hidden: (<num_layer x batch_size x hidden_size>, <num_layer x batch_size x hidden_size>)
        - lstm_out: <1 x batch_size x N_LETTER>
        """
        # input = input.view(len(input), self.batch_size, -1)
        lstm_out, hidden = self.lstm(input, hidden)
        lstm_out = self.fc1(lstm_out)
        lstm_out = self.softmax(lstm_out)
        return lstm_out, hidden

