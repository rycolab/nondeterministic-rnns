import numpy as np

from nfarnn.base.nn import id_map, l1_normalize
from nfarnn.nfarnn.elman_lm import ElmanLM
from nfarnn.nfarnn.nondeterministic_elman_transform import NondeterministicElmanTransform

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
