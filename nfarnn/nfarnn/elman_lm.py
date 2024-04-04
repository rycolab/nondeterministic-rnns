import numpy as np
from math import log
from typing import Callable

from nfarnn.base.nn import Log, softmax, id_map, l1_normalize
from nfarnn.base.symbol import Sym, EOS
from nfarnn.base.utils import to_compatible_string
from nfarnn.nfarnn.elman_network import ElmanNetwork
from nfarnn.nfarnn.nondeterministic_elman_transform import NondeterministicElmanTransform

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
        
        # print(f"h0: {self.R.h}")
        # print(f"p0: {p}")

        for i, a in enumerate(s):
            y = self.one_hot(a)
            logp += log(p[y.argmax()])

            if a == EOS:
                break
            
            self.R(y)
            p = self.π(self.F(self.R.h))

            # print(f"h{i+1}:{self.R.h}")
            # print(f"p{i+1}: {p}, y{i}: {y}")

        return logp


class SparsemaxElmanLM(ElmanLM):
    def __init__(self, R, F, one_hot) -> None:
        """ An Elman network using sparsemax as the output projection.
        
        Args:
            R (ElmanNetwork): An Elman RNN.
            F (Callable[[np.ndarray], np.ndarray]): A transformation function.
                Defaults to the identity function.
            one_hot: Callable[[Sym], np.ndarray].
        """

        # The identity function equals sparsemax for l1_normalized distributions
        super().__init__(R=R, F=F, π=id_map, one_hot=one_hot)
    
    @staticmethod
    def from_pfsa(A):
        """ Returns a SparsemaxElmanLM corresponding to a given PFSA.

        Args:
            A (FSA): A Probabilistic Finite State Automaton
        """
        M = NondeterministicElmanTransform(A)
        # F applies l1_normalziation after a linear output function
        F = lambda h: l1_normalize(M.E @ h) 
        return SparsemaxElmanLM(R=M.R, F=F, one_hot=M.one_hot)


class SoftmaxElmanLM(ElmanLM):
    def __init__(self, R, F, one_hot) -> None:
        """ An Elman network using softmax as the output projection.
        
        Args:
            R (ElmanNetwork): An Elman RNN.
            F (Callable[[np.ndarray], np.ndarray]): A transformation function.
                Defaults to the identity function.
            one_hot: Callable[[Sym], np.ndarray].
        """
        super().__init__(R=R, F=F, π=softmax, one_hot=one_hot)
        
    @staticmethod
    def from_pfsa(A):
        """ Returns a SoftmaxElmanLM corresponding to a given PFSA.

        Args:
            A (FSA): A Probabilistic Finite State Automaton
        """

        M = NondeterministicElmanTransform(A)
        # F applies pointwise ln after applying a linear output function
        F = lambda h: Log(M.E@h)
        return SoftmaxElmanLM(R=M.R, F=F, one_hot=M.one_hot)
