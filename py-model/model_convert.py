import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func

# Adjust the path to include your custom qdense implementation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'py_utils')))
from qval.qlayer import \
    qdense  # Ensure this path is correct for your project structure
from qval.qval import qval


# Define the neural network architecture
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.lay1 = nn.Linear(2, 8)
        self.lay2 = nn.Linear(8, 4)
        self.lay3 = nn.Linear(4, 1)

    def forward(self, input):
        out = func.tanh(self.lay1(input))
        out = func.tanh(self.lay2(out))
        out = self.lay3(out)
        return out

# Load the pre-trained model with weights_only=True for security
loaded_model = NN()
loaded_model.load_state_dict(torch.load('xor_model.pth', weights_only=True))

# Function to convert PyTorch layer parameters to qdense format
def convert_to_qdense_params(layer):
    input_weights = layer.weight.detach().numpy().tolist()
    output_bias = layer.bias.detach().numpy().tolist()
    return input_weights, output_bias

# Convert model parameters for qdense layers
input_weights, output_bias = convert_to_qdense_params(loaded_model.lay1)
qdense1 = qdense(input_weights, output_bias, input_bits=8, output_bits=8)
print(len(input_weights[0]))
print(qdense1)
input_weights, output_bias = convert_to_qdense_params(loaded_model.lay2)
qdense2 = qdense(input_weights, output_bias, input_bits=8, output_bits=8)

input_weights, output_bias = convert_to_qdense_params(loaded_model.lay3)
qdense3 = qdense(input_weights, output_bias, input_bits=8, output_bits=8)

# Function to process inputs through the qdense layers
def dense_process(input):
    # Ensure the input is in the correct format for qdensea
    print("LEN INPUT")
    input = input.numpy().tolist()
    print(len(input))
    print(type(input))
    out = qdense1.process(input)  # Expecting 2 inputs here
    print(f"Output of qdense1: {qval.dequantize(out[0])}")  # Should be 8
    out = qdense2.process(out)     # Expecting 8 inputs here
    print(f"Output of qdense2: {qval.dequantize(out[0])}")  # Should be 4
    out = qdense3.process(out)     # Expecting 4 inputs here
    print(f"Output of qdense3: {qval.dequantize(out[0])}")  # Should be 1
    return out

# Generate example dataset for XOR operation
datasetSize = 5
dataset = np.random.rand(datasetSize, 2)
dataset = np.rint(dataset)  # Convert to binary (0 or 1)
new_dataset = np.zeros((datasetSize, 3))
new_dataset[:, :2] = dataset
new_dataset[:, 2] = (dataset[:, 0] != dataset[:, 1])  # XOR

tensorDataset = torch.tensor(new_dataset, dtype=torch.float32)
inputs = tensorDataset[:, :2]

# Process each input through the dense layers and print outputs
for i in range(len(inputs)):
    input_row = inputs[i]
    print(f"Input: {input_row.numpy()}")  # Convert tensor to numpy array for printing
    output = dense_process(input_row)
    print(f"Output: {output}")
