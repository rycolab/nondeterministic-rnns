from pytest import mark

from math import log, isclose

from nfarnn.base.utils import sample_string
from nfarnn.base.random import random_pfsa
from nfarnn.nfarnn.nondeterministic_elman_transform import NondeterministicElmanTransform

@mark.parametrize("softmax_normalized", [False, True])
@mark.parametrize("n_states", [3, 5, 7, 9])
@mark.parametrize("alphabet_size", [2, 3, 4, 5, 6])
def test_nfa_rnn(n_states: int, alphabet_size: int, softmax_normalized: bool):
    for _ in range(50):
        A = random_pfsa(
            Sigma="abcde"[:alphabet_size],
            num_states=n_states,
            deterministic=False,
            no_eps=True,
            bias=1.0)
        
        x = sample_string(A)
        if A.accept(x).value < 1e-7:
            continue

        M = NondeterministicElmanTransform(A, softmax_normalized=softmax_normalized)

        FSA_score, RNN_score = log(A.accept(x)), M.R.score(x)

        assert isclose(FSA_score, RNN_score, rel_tol=1e-6)

test_nfa_rnn(10, 6, True)
