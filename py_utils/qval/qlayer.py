from typing import Callable

from . import qfunctions as qf
from . import qval


class qdense():
    def __init__(self, input_values: list[list], output_values: list,
                 input_bits: int = 8, output_bits: int = 8):

        self.input_len = len(input_values[0])
        self.output_len = len(output_values)
        self.input_bits = input_bits
        self.output_bits = output_bits
        self.input_params = []
        self.output_params = []

        if len(input_values) != len(output_values):
            raise ValueError("Invalid number of inputs and/or outputs.")

        if isinstance(input_values[0][0], float):
            self.input_params = [[qval(x, input_bits)
                                  for x in values] for values in input_values]
            self.output_params = [qval(x, output_bits) for x in output_values]
            print(f"INPUT PARAMS: {self.input_params}")

        elif isinstance(input_values[0][0], qval):
            self.input_params = input_values
            self.output_params = output_values

        # elif isinstance(input_values[0], str):
        #     if not all(char in '01' for char in value):
        #     raise ValueError("BinaryString can only contain '0' and '1'")


        else:
            raise ValueError("Unsuported type for qval.")

    def process(self, inputs: list) -> list:
        if len(inputs) != self.input_len:
            raise ValueError("Invalid number of inputs.")

        if isinstance(inputs[0], float):
            inputs = [qval(x, self.input_bits) for x in inputs]

        elif isinstance(inputs[0], qval):
            inputs = inputs

        else:
            raise ValueError("Unsuported type for qval.")

        output = []

        for i in range(self.output_len):
            result = qval(0.0, self.output_bits)

            for param, input in zip(self.input_params[i], inputs):
                result += param * input

            output.append(result + self.output_params[i])

        return output

    def __repr__(self):
        input_layer = f"Input layers: q{self.input_bits}\n"
        for i in range(self.input_len):
            input_layer += f"{i} {[x.val() for x in self.input_params[i]]}\n"

        output_layer = f"Output layers: q{self.output_bits} { \
            [x.val() for x in self.output_params]} \n"
        return f"{input_layer + output_layer}"
