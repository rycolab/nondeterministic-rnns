from math import log
from typing import Callable, Optional, Tuple, Union
import numpy as np

from nfarnn.base.symbol import Sym, EOS
from nfarnn.base.utils import to_compatible_string
from nfarnn.base.nn import ReLU, id_map

class ElmanNetwork:

    def __init__(
        self,
        U: np.ndarray,
        V: np.ndarray,
        b: np.ndarray,
        E: np.ndarray,
        h0: np.ndarray,
        one_hot: Callable[[Sym], np.ndarray],
        α: Callable[[np.ndarray], np.ndarray] = ReLU,
        π: Callable[[np.ndarray], np.ndarray] = id_map,
        F: Callable[[np.ndarray], np.ndarray] = id_map,
    ):
        """Implementation of an Elman network with the Heaviside non-linearity.

        DEFINITION
        A Elman language model is a tuple
        • h0 is a D-dimensional initial vector;
        • U a D x D transition matrix;
        • V a D x |Σ| symbol matrix;
        • E a |Σ| x (D+1) emission matrix;
        • b a |Σ|-dimensional bias term.
        • α is the hidden state activation function.
        • π is a projection function from the unnormalized scores to the normalized
            local probability distribution over the next symbol.

        Args:
            U (np.ndarray): The transition matrix.
            V (np.ndarray): The symbol matrix.
            b (np.ndarray): The bias term.
            E (np.ndarray): The emission matrix.
            h0 (np.ndarray): The initial hidden state.
            f (Callable[[np.ndarray], np.ndarray]): The projection function.
            one_hot (Callable[Sym, np.ndarray]): The function that maps a symbol to its
                one-hot encoding.
            α (Callable[[np.ndarray], np.ndarray], optional): The hidden state
                activation function. Defaults to ReLU.
            π (Callable[[np.ndarray], np.ndarray], optional): The output projection
                function. Defaults to the identity function.    
            F (Callable[[np.ndarray], np.ndarray], optional): A non-linear transformation
                of the RNN output.
        """

        self.U = U
        self.V = V
        self.b = b
        self.E = E
        self.h0 = h0
        self.sym_one_hot = one_hot
        self.α = α
        self.π = π
        self.F = F

    def score(self, s: str) -> float:
        s = to_compatible_string(s)

        logp = 0.0 
        h = self.h0
        p = self.π(self.F(self.E @ h))

        print(f"h0: {h}")

        for i, a in enumerate(s):
            y = self.sym_one_hot(a)
            logp += log(p[y.argmax()])

            print(f"p{i}: {p}, y{i}: {y}")

            if a == EOS:
                break
            h, p = self(h, y)
            
            print(f"h{i+1}:{h}")

        return logp

    def __call__(
        self, h: np.ndarray, y: Optional[Union[str, Sym, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if y is not None:
            if isinstance(y, str):
                y = Sym(y)

            if isinstance(y, Sym):
                y = self.sym_one_hot(y)

            h = self.α(np.dot(self.U, h) + np.dot(self.V, y) + self.b)

        p = self.π(self.F(self.E @ h))

        return h, p
