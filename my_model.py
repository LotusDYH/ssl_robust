import torch.nn as nn

class CLModel(nn.Module):

    def __init__(self, hidden_size):
        super(CLModel, self).__init__()
        self.linear = nn.Linear(hidden_size, 128)

    def forward(self, inputs):
        outputs_z = self.linear(inputs)
        outputs_z = nn.Tanh()(outputs_z)
        return outputs_z
