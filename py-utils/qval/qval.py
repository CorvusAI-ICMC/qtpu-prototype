from typing import Callable
from . import qfunctions as qf

class qval(str):
    def __new__(self, value, n: int = 8, scale: float = 1.0):
        self.scale = scale

        if isinstance(value, float):
            value_range = pow(2, n) - 2
            
            while abs(value) / self.scale > value_range / 2:
                self.scale *= 2
            
            if self.scale > 1.0:
                print(f"Redefining scale: {self.scale}")

            value /= self.scale

            return str.__new__(self, qf.quantize(value, n, self.scale))

        if not isinstance(value, str):
            raise ValueError("Unsuported type for qval.")

        # Ensure that the value contains only '0' and '1'
        if not all(char in '01' for char in value):
            raise ValueError("BinaryString can only contain '0' and '1'")
        
        # print("??", self.scale)
        self.scale = scale

        return str.__new__(self, value)

    @staticmethod
    def quantize(x: float,  n: int = 4, scale: int = 1.0) -> str:
        return qf.quantize(x, n, scale)

    @staticmethod
    def dequantize(self) -> float:
        return qf.dequantize(self)

    def val(self) -> float:
        return qf.dequantize(self)
    
    def exec(self, f:  Callable[[float], float]):
        return qf.qFun(f)(self)

    def __add__(self, other):
        return qval(qf.qAdd(self, other), len(self), self.scale)

    def __sub__(self, other):
        return qval(qf.qSub(self, other), len(self), self.scale)

    def __invert__(self):
        return qval(qf.qInv(self), len(self), self.scale)

    def __mul__(self, other):
        return qval(qf.qMul(self, other), len(self), self.scale)

    def __repr__(self):
        return f"qval(bits='{self}', size={len(self)}, decimal='{qf.qToInt(self)}', scale={self.scale})"
