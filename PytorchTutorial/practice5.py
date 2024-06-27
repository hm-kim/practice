import torch
from torch import nn


# input
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# flatten
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# linear
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}\n\n")

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)

logits = seq_modules(input_image)
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(f"Result tensor size: {pred_probab.size()}\n")
print(f"Result: {pred_probab}\n")