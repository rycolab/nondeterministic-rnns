from math import log
from typing import Callable, Optional, Tuple, Union
import numpy as np

from nfarnn.base.symbol import Sym
from nfarnn.base.utils import to_compatible_string

class ElmanNetwork:
    def ReLU(x: np.ndarray) -> np.ndarray:
        """The ReLU function."""
        return np.maximum(x, 0)

    def __init__(
        self,
        U: np.ndarray,
        V: np.ndarray,
        b: np.ndarray,
        E: np.ndarray,
        h0: np.ndarray,
        f: Callable[[np.ndarray], np.ndarray],
        one_hot: Callable[[Sym], np.ndarray],
        σ: Callable[[np.ndarray], np.ndarray] = ReLU,
        n_applications: int = 1,
    ):
        """Implementation of an Elman network with the Heaviside non-linearity.

        DEFINITION
        A Elman language model with the Heaviside non-linearity is a tuple
        • h0 is a D-dimensional initial vector;
        • U a D x D transition matrix;
        • V a D x |Σ| symbol matrix;
        • E a |Σ| x (D+1) emission matrix;
        • b a |Σ|-dimensional bias term.
        • f is a projection function from the unnormalized scores to the normalized
            local probability distribution over the next symbol.
        • σ is the hidden state activation function.

        Args:
            U (np.ndarray): The transition matrix.
            V (np.ndarray): The symbol matrix.
            b (np.ndarray): The bias term.
            E (np.ndarray): The emission matrix.
            h0 (np.ndarray): The initial hidden state.
            f (Callable[[np.ndarray], np.ndarray]): The projection function.
            σ (Callable[[np.ndarray], np.ndarray], optional): The hidden state
                activation function. Defaults to H (Heaviside).
            one_hot (Callable[Sym, np.ndarray]): The function that maps a symbol to its
                one-hot encoding.
            n_applications (int, optional): The number of times the Elman update step
                is applied to the hidden state.
        """

        self.U = U
        self.V = V
        self.b = b
        self.E = E
        self.h0 = h0
        self.f = f
        self.sym_one_hot = one_hot
        self.σ = σ
        self.n_applications = n_applications

    def score(self, s: str) -> float:
        s = to_compatible_string(s)

        logp = 0.0 
        h = self.h0
        p = self.E @ h
        # print(f"h0: {h}")

        for i, a in enumerate(s):
            y = self.sym_one_hot(a)
            # print(f"p{i}: {p}, y{i}: {y}")
            
            logp += log(p[y.argmax()])

            h, p = self(h, y)
            # print(f"h{i+1}:{h}")

        return logp

    def __call__(
        self, h: np.ndarray, y: Optional[Union[str, Sym, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if y is not None:
            if isinstance(y, str):
                y = Sym(y)

            if isinstance(y, Sym):
                y = self.sym_one_hot(y)

            h = self.σ(np.dot(self.U, h) + np.dot(self.V, y) + self.b)

        p = self.E @ h
        p = p /  np.linalg.norm(p, 1, axis=0, keepdims=True) if np.any(p) else p

        return h, p
