import numpy as np
from math import log, inf
from typing import Callable, Set

from nfarnn.base.symbol import Sym, EOS
from nfarnn.base.utils import to_compatible_string
from nfarnn.nfarnn.elman_network import ElmanNetwork

class ElmanLM:
    """Implementation of an Elman LM.

        DEFINITION
        An Elman LM is a tuple
        • R is an Elman RNN.
        • π is a projection function from the unnormalized scores to the 
            normalized local probability distribution over the next symbol.
        • F is a (potentially non-linear) transformation of the RNN output.

        Args:
            R (ElmanNetwork): An Elman RNN.
            F (Callable[[np.ndarray], np.ndarray]): A transformation function. 
                Defaults to the identity function.    
            π (Callable[[np.ndarray], np.ndarray]): The output projection
                of the RNN output.
            one_hot: Callable[[Sym], np.ndarray].
    """
    def __init__(self, 
                 R: ElmanNetwork, 
                 F: Callable[[np.ndarray], np.ndarray],
                 π: Callable[[np.ndarray], np.ndarray],
                 one_hot: Callable[[Sym], np.ndarray]) -> None:
        self.R = R
        self.π = π
        self.F = F
        self.one_hot=one_hot

    def score(self, s: str) -> float:
        s = to_compatible_string(s)

        logp = 0.0 
        self.R.reset()

        p = self.π(self.F(self.R.h))
        
        print(f"h0: {self.R.h}")
        print(f"p0: {p}")

        for i, a in enumerate(s):
            y = self.one_hot(a)
            logp += log(p[y.argmax()])

            if a == EOS:
                break
            
            self.R(y)
            p = self.π(self.F(self.R.h))

            print(f"h{i+1}:{self.R.h}")
            print(f"p{i+1}: {p}, y{i}: {y}")

        return logp
    