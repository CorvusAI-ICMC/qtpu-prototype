import math
from typing import Callable



class qval(str):
    def __new__(cls, value, n):
        # Ensure that the value contains only '0' and '1'
        if not all(char in '01' for char in value):
            raise ValueError("BinaryString can only contain '0' and '1'")
        return str.__new__(cls, value)

def qNaN(n : int) -> str:
    """
    Returns the Not a Number (NaN) or error (E) representation for a given number of bits.

    Args:
        n (int): The number of bits for the qNaN representation.

    Returns:
        str: The qNaN representation.

    """
    return '1' + '0' * (n - 1)

def qZERO(n : int) -> str:
    """
    Returns the zero representation for a given number of bits.

    Args:
        n (int): The number of bits for the qZERO representation.

    Returns:
        str: The qZERO representation.

    """
    return '0' * n

def qMin(n : int) -> str:
    """
    Returns the minimum quantized value representation for a given number of bits.

    Args:
        n (int): The number of bits for the qMin representation.

    Returns:
        str: The qMin representation.

    """
    return '1' + '1' * (n - 1)

def qMax(n : int) -> str:
    """
    Returns the maximum quantized value representation for a given number of bits.

    Args:
        n (int): The number of bits for the qMax representation.

    Returns:
        str: The qMax representation.

    """
    return '0' + '1' * (n - 1)

def qMax_i(n : int) -> int:
    """
    Returns the int module of the maximun value for a representation witha given
    number of bits.

    Args:
        n (int): The number of bits for the representation.

    Returns:
        int: The maximum value for the representation.
            
    """
    
    return 2 ** (n - 1) - 1

def qStep(n : int) -> float:
    """
    Returns the step size for a given number of bits.

    Args:
        n (int): The number of bits for the step size.

    Returns:
        float: The step size.

    """
    return 1.0 / (2**(n - 1) - 1)

def qPlanck(n : int) -> float:
    """
    Returns the minimum difference value for quantization for a given number of bits.

    Args:
        n (int): The number of bits for the quantization.

    Returns:
        float: The minimum difference value for quantization.

    """

    return qStep(n) / 2.0


def quantize(x: float, n: int = 4) -> str:
    """
    Quantizes a given floating-point number `x` into a binary string representation
    with `n` bits using our method.

    Args:
        x (float): The input floating-point number to be quantized.
        n (int, optional): The number of bits for the quantized representation. Defaults to 4.

    Returns:
        str: The quantized binary string representation of `x`.

    Raises:
        None

    Notes:
        - The function assumes that -1.0 <= x <= 1.0 and n >= 2.
        - The function uses the following special binary strings for specific cases:
            - qNaN: '0b1' + '0' * (n - 1) (NaN in quantized form, representing -0)
            - qZERO: '0b0' + '0' * (n - 1) (zero value, representing +0)
            - qMin: '0b1' + '1' * (n - 1) (min quantized value, representing -inf)
            - qMax: '0b0' + '1' * (n - 1) (max quantized value, representing +inf)
        - The function uses a step size and a minimum difference value for quantization.
        - The function rounds the quantized value to the nearest integer.

    Examples:
        >>> quantize_s(0.5, 4)
        '0100'
        >>> quantize_s(-0.75, 4)
        '1010'
    """


    # Especial cases
    if math.isnan(x): return qNaN(n)
    elif math.isinf(x): return qNaN(n)
    elif x <= -1.0: return qMin(n)
    elif x >= 1.0: return qMax(n)
    elif abs(x) <= qPlanck(n): return qZERO(n)

    # Defining the sign bit
    sign_bit = '0' if x >= 0 else '1'

    # From now on, lets work with the absolute value of x
    x = abs(x) 

    # Quantizing the value
    q_val = x * float(qMax_i(n))

    # Rounding the value to the nearest quantized value
    q_val = round(q_val)
    
    # Adding the sign bit and converting to binary
    result = sign_bit + bin(q_val)[2:].zfill(n - 1)
    return result


def dequantize(q: str) -> float:
    n = len(q)

    if n < 1:
        raise ValueError('The number of bits must be an integer greater than zero.')

    if q == qNaN(n): return float('nan')
        
    sign = -1.0 if q[0] == '1' else 1.0

    val = float(int(q[1:], 2)) / float(qMax_i(n))

    return sign * val


def qToInt(q : str) -> int:
    if len(q) < 1:
        raise ValueError('The number of bits must be an integer greater than zero.')
    
    val = int(q[1:], 2)
    return -val if q[0] == '1' else val


def qfit(q: str, n : int) -> str:
    if n < 1: 
        raise ValueError('The number of bits must be an integer greater than zero.')
    
    return q[:n] if len(q) > n else q.ljust(n, '0')


def qFromInt(x : int, n : int) -> str:
    if n < 1:
        raise ValueError('The number of bits must be an integer greater than zero.')
    
    abs_val = abs(x)
    sign_bit = '1' if x < 0 else '0'
    result = sign_bit + bin(abs_val)[2:].zfill(n - 1)

    return result if len(result) <= n else qMax(n)
         

def qAdd(a : str, b : str) -> str:
    if len(a) != len(b): 
        raise ValueError('Both quantized values must have the same number of bits.')
    if len(a) < 1:
        raise ValueError('The number of bits must be an integer greater than zero.')
    
    return qFromInt(qToInt(a) + qToInt(b), len(a))


def qSub(a : str, b : str) -> str:
    if len(a) != len(b): 
        raise ValueError('Both quantized values must have the same number of bits.')
    if len(a) < 1:
        raise ValueError('The number of bits must be an integer greater than zero.')
    
    return qFromInt(qToInt(a) - qToInt(b), len(a) + 1)


def qInv(q : str) -> str:
    if len(q) < 1:
        raise ValueError('The number of bits must be an integer greater than zero.')
    return ('1' if '0' == q[0] else '0') + q[1:]


def qMul(a : str, b : str) -> str:
    if len(a) != len(b): 
        raise ValueError('Both quantized values must have the same number of bits.')
    if len(a) < 1:
        raise ValueError('The number of bits must be an integer greater than zero.')
    
    val = qToInt(a) * qToInt(b)
    return qFromInt(val, 1 + (len(a) - 1) * 2)

def qFun(f : Callable[[float], float]) -> str:
    return lambda q: quantize(f(dequantize(q)), len(q))


