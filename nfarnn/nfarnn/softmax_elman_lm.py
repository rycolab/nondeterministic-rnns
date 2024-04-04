from nfarnn.base.nn import Log, softmax
from nfarnn.nfarnn.elman_lm import ElmanLM
from nfarnn.nfarnn.nondeterministic_elman_transform import NondeterministicElmanTransform

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
