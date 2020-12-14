import bigfloat
from comb_and_prob_funcs import *

# Uses a uniform prior for the not-null.
#
# S -- number of flips
# C -- number of flips that landed heads
# p_null -- the proportion of flips that the null hypothesis expects to be heads
def bayes_factor_for_binomial(S, C, p_null):
    p_given_null = bigfloat_prob_of_count_given_p(C, p_null, S)
    # It turns out that a uniform prior for the coin prob means a uniform
    # guess about the count.
    p_given_not_null = bigfloat.BigFloat(1.0) / (S + 1.0)
    return p_given_not_null / p_given_null
