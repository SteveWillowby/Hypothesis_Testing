import bigfloat
from comb_and_prob_funcs import bigfloat_prob_of_count_given_p

# S -- number of flips
# C -- number of flips that landed heads
# p_null -- the proportion of flips that the null hypothesis expects to be heads
def pvalue_for_binomial(S, C, p_null):
    if C >= p_null * S:
        C = S - C
        p_null = 1.0 - p_null

    total_prob = bigfloat.BigFloat(0.0)
    for i in range(0, int(C) + 1):
        total_prob += bigfloat_prob_of_count_given_p(i, p_null, S)

    return total_prob * 2.0

def approx_pvalue_for_binomial(S, C, p_null):
    print("approx_pvalue_for_binomial() is not implemented!")
    exit(1)

__precalc_pvalue_cache__ = {}
__precalc_pvalue_S__ = 0.0
__precalc_pvalue_p_null__ = 0.0

# S -- number of flips
# C -- number of flips that landed heads
# p_null -- the proportion of flips that the null hypothesis expects to be heads
def precalc_pvalue_for_binomial(S, C, p_null):
    assert int(C) == C
    C = int(C)

    global __precalc_pvalue_cache__
    global __precalc_pvalue_S__
    global __precalc_pvalue_p_null__

    if __precalc_pvalue_S__ != S or __precalc_pvalue_p_null__ != p_null:
        __precalc_pvalue_S__ = S
        __precalc_pvalue_p_null__ = p_null
        __precalc_pvalue_cache__ = {}

        most_likely_C = int(round(float(p_null * S)))

        total_prob = bigfloat.BigFloat(0.0)
        for C_prime in range(0, most_likely_C + 1):
            total_prob += bigfloat_prob_of_count_given_p(C_prime, p_null, S)
            __precalc_pvalue_cache__[C_prime] = bigfloat.min(total_prob * 2.0, 1.0)

        total_prob = bigfloat.BigFloat(0.0)
        for i in range(0, int(S) - most_likely_C):
            C_prime = S - i
            total_prob += bigfloat_prob_of_count_given_p(C_prime, p_null, S)
            __precalc_pvalue_cache__[C_prime] = bigfloat.min(total_prob * 2.0, 1.0)

    return __precalc_pvalue_cache__[C]
