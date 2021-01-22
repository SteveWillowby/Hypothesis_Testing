from optimizing import *
from comb_and_prob_funcs import bigfloat_prob_of_count_given_p

def __best_m__(P_X_given_N, P_N, n):
    assert P_N != 1.0
    assert P_N != 0.0
    P_not_N = 1.0 - P_N

    best_m_numerator = 1.0 + bigfloat.sqrt(1.0 + (P_not_N / P_N) * (1.0 + 1.0 / (P_X_given_N * P_N * (n - 1))))
    best_m_denominator = P_X_given_N + 1.0 / (P_N * (n - 1))
    return bigfloat.max(n, best_m_numerator / best_m_denominator)

def my_universal_bound(P_X_given_N, P_N, n):
    assert P_N != 1.0
    assert P_N != 0.0
    P_not_N = 1.0 - P_N
    best_m = __best_m__(P_X_given_N, P_N, n)

    chance_correct = (P_not_N * (1.0 - (n - 1) / best_m))
    if_correct = (P_X_given_N * P_N) / (P_X_given_N * P_N + (1.0 / best_m) * P_not_N)
    p_star = chance_correct * if_correct + (1.0 - chance_correct) * P_N
    return ((1.0 - p_star) * P_N) / (p_star * P_not_N)

def my_universal_other_bound(P_X_given_N, P_N, n):
    assert P_N != 1.0
    assert P_N != 0.0
    P_not_N = 1.0 - P_N
    best_m = __best_m__(P_X_given_N, P_N, n)

    chance_correct = (P_not_N * (1.0 - (n - 1) / best_m))
    if_correct = (P_X_given_N * P_N) / (P_X_given_N * P_N + (1.0 / best_m) * P_not_N)
    p_star = chance_correct * if_correct + (1.0 - chance_correct) * P_N
    return (P_N / p_star) # * ((1.0 - p_star) / P_not_N)

# S -- number of flips
# C -- number of flips that landed heads
# p_null -- the proportion of flips that the null hypothesis expects to be heads
# P_N == prior probability that the null is true
def my_universal_bound_on_binomial(S, C, p_null, P_N=0.5):
    n = S + 1
    P_X_given_N = bigfloat_prob_of_count_given_p(C, p_null, S)

    return my_universal_bound(P_X_given_N, P_N, n)

__memoized_my_universal_bound_values__ = {}

# S -- number of flips
# C -- number of flips that landed heads
# p_null -- the proportion of flips that the null hypothesis expects to be heads
# P_N == prior probability that the null is true
def memoized_my_universal_bound_on_binomial(S, C, p_null, P_N=0.5):
    assert int(C) == C
    C = int(C)
    assert int(S) == S
    S = int(S)
    assert float(P_N) == P_N
    P_N = float(P_N)
    assert float(p_null) == p_null
    p_null = float(p_null)

    global __memoized_my_universal_bound_values__

    key = (S, C, p_null, P_N)
    if key in __memoized_my_universal_bound_values__:
        return __memoized_my_universal_bound_values__[key]

    value = my_universal_bound_on_binomial(S, C, p_null, P_N)
    __memoized_my_universal_bound_values__[key] = value
    return value

__memoized_my_binomial_bound_on_values__ = {}

# Returns the confidence that p_given_p >= b
def __my_binomial_bound_on_binomial_helper_helper__(b, p, S):
    total = bigfloat.BigFloat(0.0)
    for C in range(0, int(S * p) + 1):
        p_given_p = bigfloat_prob_of_count_given_p(C, p, S)
        if p_given_p >= b:
            break
        total += p_given_p
    for i in range(0, int(S) - int(S*p)):
        C = int(S) - i
        p_given_p = bigfloat_prob_of_count_given_p(C, p, S)
        if p_given_p >= b:
            break
        total += p_given_p
    return 1.0 - total

def __my_binomial_bound_on_binomial_helper__(P_X_given_N, b, S, P_N):
    func = (lambda x: lambda y: __my_binomial_bound_on_binomial_helper_helper__(x[0], y, x[1]))((b, S))
    (best_p, lowest_outer_confidence) = binary_min_finder(func, 0.0, 1.0)
    print("Best p: %f" % best_p)

    P_not_N = 1.0 - P_N
    chance_correct = lowest_outer_confidence
    if_correct = (P_X_given_N * P_N) / (P_X_given_N * P_N + b * P_not_N)
    alpha = chance_correct * if_correct + (1.0 - chance_correct) * P_N

    return ((1.0 - alpha) * P_N) / (alpha * P_not_N)

# Makes strong convexity assumption on both p and b.
def my_binomial_bound_on_binomial(S, C, coin_prob, P_N=0.5):
    P_X_given_N = bigfloat_prob_of_count_given_p(C, coin_prob, S)
    func = (lambda x: lambda y: -1.0 * __my_binomial_bound_on_binomial_helper__(x[0], y, x[1], x[2]))((P_X_given_N, S, P_N))
    (best_b, neg_evidence) = binary_min_finder(func, 0.0, 1.0)
    print("Best b: %f" % best_b)
    return -1.0 * neg_evidence

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    outer_values = []
    prob_values = []
    for numerator in range(0, 10000):
        prob_value = numerator / 10000.0
        outer_value = __my_binomial_bound_on_binomial_helper_helper__(0.1, prob_value, 10.0)
        outer_values.append(outer_value)
        prob_values.append(prob_value)

    plt.plot(prob_values, outer_values)
    plt.show()
