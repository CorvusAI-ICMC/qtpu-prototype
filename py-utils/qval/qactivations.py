from . import qfunctions as qf
import numpy as np

def sigmoid(x: float) -> float:
    """
    Returns the sigmoid function for a given input x.
    
    Args:
        x (float): The input value.
        
    Returns:
        float: The sigmoid function value.
        
    """
    return 1 / (1 + np.exp(-x))

def tanh(x: float) -> float:
    """
    Returns the hyperbolic tangent function for a given input x.
    
    Args:
        x (float): The input value.
        
    Returns:
        float: The hyperbolic tangent function value.
        
    """
    return np.tanh(x)

def relu(x: float) -> float:
    """
    Returns the rectified linear unit function for a given input x.
    
    Args:
        x (float): The input value.
        
    Returns:
        float: The rectified linear unit function value.
        
    """
    return max(0, x)

def leaky_relu(x: float, alpha: float = 0.01) -> float:
    """
    Returns the leaky rectified linear unit function for a given input x.
    
    Args:
        x (float): The input value.
        alpha (float): The slope for the negative values.
        
    Returns:
        float: The leaky rectified linear unit function value.
        
    """
    return max(alpha * x, x)

def elu(x: float, alpha: float = 1.0) -> float:
    """
    Returns the exponential linear unit function for a given input x.
    
    Args:
        x (float): The input value.
        alpha (float): The slope for the negative values.
        
    Returns:
        float: The exponential linear unit function value.
        
    """
    return x if x > 0 else alpha * (np.exp(x) - 1)

def softplus(x: float) -> float:
    """
    Returns the softplus function for a given input x.
    
    Args:
        x (float): The input value.
        
    Returns:
        float: The softplus function value.
        
    """
    return np.log(1 + np.exp(x))

def swish(x: float) -> float:
    """
    Returns the swish function for a given input x.
    
    Args:
        x (float): The input value.
        
    Returns:
        float: The swish function value.
        
    """
    return x * sigmoid(x)






qSigmoid = qf.qFun(sigmoid)
qTanh = qf.qFun(tanh)
qRelu = qf.qFun(relu)
qLeakyRelu = qf.qFun(leaky_relu)
qElu = qf.qFun(elu)
qSoftplus = qf.qFun(softplus)
qSwish = qf.qFun(swish)