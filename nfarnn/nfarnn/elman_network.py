import numpy as np
from typing import Callable, Tuple
from nfarnn.base.nn import ReLU

class ElmanNetwork:

    def __init__(
        self,
        h0: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        b: np.ndarray,
        α: Callable[[np.ndarray], np.ndarray] = ReLU
    ):
        """Implementation of an Elman network.

        DEFINITION
        A Elman RNN is a tuple
        • h0 is a D-dimensional initial vector;
        • U a D x D transition matrix;
        • V a D x |Σ| symbol matrix;
        • b a |Σ|-dimensional bias term.
        • α is the hidden state activation function.

        Args:
            h0 (np.ndarray): The initial hidden state.
            U (np.ndarray): The transition matrix.
            V (np.ndarray): The symbol matrix.
            b (np.ndarray): The bias term.
            α (Callable[[np.ndarray], np.ndarray], optional): The hidden state
                activation function. Defaults to ReLU.
        """

        self.h0 = h0
        self.h = h0
        self.U = U
        self.V = V
        self.b = b
        self.α = α

    def reset(self):
        self.h = self.h0

    def __call__(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.h = self.α(np.dot(self.U, self.h) + np.dot(self.V, y) + self.b)
