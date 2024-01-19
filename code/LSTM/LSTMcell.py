import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias):
        super(LSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bais = bias
        self.x2h = nn.Linear()
        self.h2h = nn.Linear()

        self.reset_parameters()

    def reset_parameters(self):
        ...

    def forward(self, x, hidden):
        hx, cx = hidden

        gates = self.x2h(x) + self.h2h(hx)
        gates.squeeze()
        inputgate, forgetgate, cellgate, outputgate = gates.chunk(4,1)

        inputgate = F.sigmoid(inputgate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outputgate = F.sigmoid(outputgate)

        c_t = torch.mul(forgetgate, cx) + torch.mul(inputgate, cellgate)
        output = torch.mul(c_t, F.tanh(c_t))

        return output
        
    
    

