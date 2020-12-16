from optimizing import *
from comb_and_prob_funcs import bigfloat_prob_of_count_given_p

def __best_m_and_evidence_strength_helper__(P_X_given_N, P_N, n, m):
    assert P_N != 1.0
    assert P_N != 0.0
    P_not_N = 1.0 - P_N
    chance_correct = (P_not_N * (1.0 - (n - 1) / m))
    if_correct = (P_X_given_N * P_N) / (P_X_given_N * P_N + (1.0 / m) * P_not_N)
    alpha = chance_correct * if_correct + (1.0 - chance_correct) * P_N
    return ((1.0 - alpha) * P_N) / (alpha * P_not_N)

# S -- number of flips
# C -- number of flips that landed heads
# p_null -- the proportion of flips that the null hypothesis expects to be heads
def my_bound_for_binomial(S, C, coin_prob, P_N=0.5):
    n = S + 1
    P_X_given_N = bigfloat_prob_of_count_given_p(C, coin_prob, S)
    # if P_X_given_N >= 1.0 / n:
    #     return 1.0
    func = (lambda x: lambda y: __best_m_and_evidence_strength_helper__(x[0], x[1], x[2], y))((P_X_given_N, P_N, n))
    # return (best_arg, best_func)
    (best_m, best_func_val) = max_finder_low_bound_only(func, S)

    return best_func_val

__memoized_my_bound_values__ = {}
__memoized_my_bound_S__ = 0.0
__memoized_my_bound_coin_prob__ = 0.0
__memoized_my_bound_P_N__ = 0.0

# S -- number of flips
# C -- number of flips that landed heads
# p_null -- the proportion of flips that the null hypothesis expects to be heads
def memoized_my_bound_for_binomial(S, C, coin_prob, P_N=0.5):
    assert int(C) == C
    C = int(C)

    global __memoized_my_bound_values__
    global __memoized_my_bound_S__
    global __memoized_my_bound_coin_prob__
    global __memoized_my_bound_P_N__

    if __memoized_my_bound_S__ != S or __memoized_my_bound_coin_prob__ != coin_prob or \
             __memoized_my_bound_P_N__ != P_N:
        # COMPUTE
        pass

    # return
