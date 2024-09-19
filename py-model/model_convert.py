import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'py_utils')))
from qval.qactivations import qSigmoid, qTanh
from qval.qlayer import \
    qdense  
from qval.qval import qval



class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.lay1 = nn.Linear(2, 8)
        self.lay2 = nn.Linear(8, 4)
        self.lay3 = nn.Linear(4, 1)
    
    def forward(self, input):
        out = func.tanh(self.lay1(input))
        out = func.tanh(self.lay2(out))
        out = func.sigmoid(self.lay3(out))
        return out


loaded_model = NN()
loaded_model.load_state_dict(torch.load('xor_model.pth', weights_only=True))

for name, param in loaded_model.named_parameters():
    print(f"{name}: {param.data}")

test_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
with torch.no_grad():
    outputs = loaded_model(test_inputs)
print("XOR inputs:", test_inputs)
print("Model outputs:", outputs)


def convert_to_qdense_params(layer):
    input_weights = layer.weight.detach().numpy().tolist()
    print("input weights: ")
    print(np.shape(input_weights))
    output_bias = layer.bias.detach().numpy().tolist()
    return input_weights, output_bias


input_weights, output_bias = convert_to_qdense_params(loaded_model.lay1)
qdense1 = qdense(input_weights, output_bias, input_bits=8, output_bits=8)
# print(len(input_weights[0]))
# print(qdense1)
input_weights, output_bias = convert_to_qdense_params(loaded_model.lay2)
qdense2 = qdense(input_weights, output_bias, input_bits=8, output_bits=8)

input_weights, output_bias = convert_to_qdense_params(loaded_model.lay3)
qdense3 = qdense(input_weights, output_bias, input_bits=8, output_bits=8)


def dense_process(input):
    print("LEN INPUT")
    input = input.numpy().tolist()
    print(len(input))
    print(type(input))

    out = qdense1.process(input)     
    print(np.shape(out))
    print(f"Output of qdense1 (before activation): {qval.dequantize(out[0])}")  
    for i in range(len(out)):
        out[i] = qTanh(out[i])
    print(f"Output of qdense1: {qval.dequantize(out[0])}")  

    out = qdense2.process(out)  
    print(f"Output of qdense2 (before activation): {qval.dequantize(out[0])}")
    for i in range(len(out)):
        out[i] = qTanh(out[i])
    print(f"Output of qdense2: {qval.dequantize(out[0])}")  

    out = qdense3.process(out)
    print(f"Output of qdense3(before activation): {qval.dequantize(out[0])}")  
    for i in range(len(out)):
        out[i] = qSigmoid(out[i])
    print(f"Output of qdense3: {qval.dequantize(out[0])}")

    return out

datasetSize = 5
dataset = np.random.rand(datasetSize, 2)
dataset = np.rint(dataset) 
new_dataset = np.zeros((datasetSize, 3))
new_dataset[:, :2] = dataset
new_dataset[:, 2] = (dataset[:, 0] != dataset[:, 1])  

tensorDataset = torch.tensor(new_dataset, dtype=torch.float32)
inputs = tensorDataset[:, :2]

for i in range(len(inputs)):
    input_row = inputs[i]
    print(f"Input: {input_row.numpy()}") 
    output = dense_process(input_row)
    float_output = loaded_model.forward(input_row)
    print(f"Output: {qval.dequantize(output[0])}")
    print(f"float Output: {float_output}")
