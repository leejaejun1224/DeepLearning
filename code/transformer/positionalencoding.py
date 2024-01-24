import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, position, d_model):
        super(PositionalEncoding,self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)



    def get_angle_component(self, position, i, d_model):
        angles = torch.arange(1/torch.pow(10000*((2*i)//2)/torch.tensor(d_model ))
        return position * angles


    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angle_component(position = torch.arange(position, dtype=torch.float), 
                                         i = torch.arange(d_model, dtype=torch.float), 
                                         d_model=d_model)
        return angle_rads

pos = 50
d_model = 120

array = PositionalEncoding(pos, d_model)

print(array.positional_encoding(pos, d_model))


