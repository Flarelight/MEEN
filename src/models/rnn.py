import torch
import torch.nn as nn
from torch.nn import RNN



# rnn = nn.RNN(10, 20, 2)
# input = torch.randn(5, 3, 10)
# h0 = torch.randn(2, 3, 20)
# output, hn = rnn(input, h0)  # [5, 3, 20]

rnn = nn.RNN(2, 1, 4)
input = torch.randn(5, 2)

print(input.unsqueeze_(1).shape)  # (5,1,2)

h0 = torch.randn(4, 1, 1)
output, hn = rnn(input, None)

print(output.shape)  # [5, 3, 1]
print(output.squeeze_(1).shape)
print(output)




