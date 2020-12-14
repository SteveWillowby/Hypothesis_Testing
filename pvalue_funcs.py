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
        total_prob += bigfloat_prob_of_count_given_p(C, p_null, S)

    return total_prob * 2.0

def approx_pvalue_for_binomial(S, C, p_null):
    print("approx_pvalue_for_binomial() is not implemented!")
    exit(1)
