import numpy as np

def ReLU(x: np.ndarray) -> np.ndarray:
    """The ReLU activation function."""
    return np.maximum(x, 0)
    
def Log(x: np.ndarray) -> np.ndarray:
    """The log activation function."""
    with np.errstate(divide='ignore'):
        return np.log(x)

def softmax(x: np.ndarray) -> np.ndarray:
    """The softmax projection."""
    return np.exp(x)/np.sum(np.exp(x))

def id_map(x: np.ndarray) -> np.ndarray:
    """The identity function."""
    return x

def l1_normalize(x: np.ndarray) -> np.ndarray:
    """L1 normalization"""
    return x / np.linalg.norm(x, 1, axis=0, keepdims=True)
