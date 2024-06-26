from itertools import product
from typing import List, Sequence, Tuple, Union

import numpy as np

from nfarnn.base.semiring import Real
from nfarnn.base.state import State
from nfarnn.base.symbol import EOS, Sym
from nfarnn.base.fsa import FSA
from nfarnn.nfarnn.elman_network import ElmanNetwork
from nfarnn.base.nn import ReLU


class NondeterministicElmanTransform:
    def __init__(self, A: FSA, softmax_normalized=False) -> None:
        """The class performing the transformation of an FSA into a ReLU Elman RNN.

        Args:
            A (FSA): The PFSA to be transformed.
        """
        assert A.R == Real

        self.A = A
        self.Q = list(self.A.Q)
        self.Sigma = sorted(list(self.A.Sigma))
        self.SigmaEOS = self.Sigma + [EOS]
        self.n_states, self.n_symbols = len(self.Q), len(self.Sigma)

        self.D = self.n_states * self.n_symbols
        self.softmax_normalized=softmax_normalized

        self._construct()

    def initial_state(self) -> np.ndarray:
        h = np.zeros((self.D))

        for q, w in self.A.I:
            a = self.m_inv[0]
            h[self.n[(a, q)]] = float(w)

        return h
    
    def one_hot(self, x: Union[str, Sym]) -> np.ndarray:
        if isinstance(x, str):
            x = Sym(x)
        if not x in self.SigmaEOS:
            raise ValueError(f"Unknown symbol {x}")
        y = np.zeros((self.n_symbols + 1))
        y[self.m[x]] = 1
        return y

    def _set_up_orderings(self):
        self.s = dict()
        self.s_inv = dict()
        for i, q in enumerate(self.Q):
            self.s[q] = i
            self.s_inv[i] = q

        self.n = dict()
        self.n_inv = dict()
        for i, (y, q) in enumerate(product(self.SigmaEOS, self.A.Q)):
            self.n[(y, q)] = i
            self.n_inv[i] = (y, q)

        self.m = {y: i for i, y in enumerate(self.SigmaEOS)}
        self.m_inv = {i: y for i, y in enumerate(self.SigmaEOS)}

    def _construct(self):
        self._set_up_orderings()

        self._make_U()
        self._make_V()
        self._make_b()

        self._make_E()

        self._make_rnn()

    def Q2states(self, Q: Sequence[int]) -> List[State]:
        return [self.Q[q_ix] for (q_ix, q) in enumerate(Q) if q == 1]

    def _make_U(self):
        # Create new transition matrices that are normalized over states AND symbols
        T = {y: np.zeros((self.n_states, self.n_states)) for y in self.Sigma}
        for q in self.Q:
            for y, qʼ, w in self.A.arcs(q):
                T[y][(self.s[qʼ], self.s[q])] = float(w)
        
        self.U = np.zeros((self.D, self.D))
        for q in self.Q:
            for y, qʼ, w in self.A.arcs(q):
                for yʼ in self.Sigma:
                    self.U[self.n[(y, qʼ)], self.n[(yʼ, q)]] = T[y][(self.s[qʼ], self.s[q])]

    def _make_V(self):
        self.V = np.zeros((self.D, self.n_symbols+1))
        for q in self.A.Q:
            for y in self.Sigma:
                # The active copied probabilities from Step 1 will be retained
                # That is, this *undoes* the -1 from the bias vector and simply
                # retains the entries in the hidden state
                self.V[self.n[(y, q)], self.m[y]] = 1

    def _make_b(self):
        # The non-active copied probabilities from Step 1 will be removed
        # This acts as a selector of the correct transition for the observed symbol
        # (this relies on the largest possible weight being 1)
        self.b = -np.ones(self.D)

    def _make_E(self):
        self.E = np.zeros((self.n_symbols + 1, self.D))
        for q in self.Q:
            for y, _, w in self.A.arcs(q):
                for yʼ in self.Sigma:
                    self.E[self.m[y], self.n[(yʼ, q)]] += float(w)

        for q, w in self.A.F:
            for yʼ in self.Sigma:
                self.E[self.m[EOS], self.n[(yʼ, q)]] = float(w)

    def _make_rnn(self):
        self.R = ElmanNetwork(
            h0=self.initial_state(),
            U=self.U,
            V=self.V,
            b=self.b
        )
