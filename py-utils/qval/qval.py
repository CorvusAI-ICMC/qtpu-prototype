from typing import Callable
from . import qfunctions as qf

class qval(str):
    def __new__(self, value, n : int = 8):
        if isinstance(value, float):
            return str.__new__(self, qf.quantize(value, n))
        
        if not isinstance(value, str):
            raise ValueError("Unsuported type for qval.")

        # Ensure that the value contains only '0' and '1'
        if not all(char in '01' for char in value):
            raise ValueError("BinaryString can only contain '0' and '1'")

        return str.__new__(self, value)
    
    @staticmethod
    def quantize(x : float,  n: int = 4) -> str:
        return qf.quantize(x, n)
    
    @staticmethod
    def dequantize(self) -> float:
        return qf.dequantize(self)

    def val(self) -> float:
        return qf.dequantize(self)
    
    def exec(self, f:  Callable[[float], float]):
        return qf.qFun(f)(self)

    def __add__(self, other):
        return qval(qf.qAdd(self, other))
    
    def __sub__(self, other):
        return qval(qf.qSub(self, other))
    
    def __invert__(self):
        return qval(qf.qInv(self))
    
    def __mul__(self, other):
        return qval(qf.qMul(self, other))
    
    def __repr__(self):
        return f"qval(q='{self}', n={len(self)}, d='{int(self, 2)}', i='{qf.qToInt(self)}')"