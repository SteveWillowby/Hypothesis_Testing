from optimizing import *
from comb_and_prob_funcs import bigfloat_prob_of_count_given_p, bigfloat_choose
from algorithmic_utils import binary_search_for_last_meeting_criterion

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
    # if_correct_B = P_N * (P_X_given_N * best_m) / (P_not_N + P_N * (P_X_given_N * best_m))
    # print(bigfloat.abs(if_correct - if_correct_B) < if_correct)
    p_star = chance_correct * if_correct + (1.0 - chance_correct) * P_N
    return ((1.0 - p_star) * P_N) / (p_star * P_not_N)

def my_kind_of_bound_max(P_N):
    assert P_N != 1.0
    assert P_N != 0.0
    P_not_N = 1.0 - P_N

    value = (1.0 - P_N * P_N) / (P_N * P_not_N)
    # print("Max with P_N = %f: %f" % (float(P_N), float(value)))
    return value

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

__memoized_choose_values__ = {}

__precomputed_cumulative_probs__ = {}

# p -- coin prob
# C -- number of heads
# S -- total number of flips
def __alternate_bigfloat_binomial_prob__(C, p, S):
    assert p > bigfloat.BigFloat(0.0)
    assert p < bigfloat.BigFloat(1.0)

    choose_key = (int(S), int(C))
    if choose_key in __memoized_choose_values__:
        choose_val = __memoized_choose_values__[choose_key]
    else:
        choose_val = bigfloat_choose(S, C)
        __memoized_choose_values__[choose_key] = choose_val

    return choose_val * bigfloat.pow(p, C) * bigfloat.pow(1.0 - p, S - C)

# p -- coin prob
# S -- total number of flips
def __precompute_cumulative_probs_for_binomial__(p, S):
    print("Precomputing cumulative binomial probs for (%f, %d))" % (p, S))
    probs = []
    for C in range(0, int(S) + 1):
        probs.append(__alternate_bigfloat_binomial_prob__(C, p, S))

    idx_A = len(probs) - 1
    inc_A = -1
    idx_B = 0
    inc_B = 0
    cumulative_prob = bigfloat.BigFloat(1.0)
    # Stores pairs (p1, p2) where p2 is meta-prob that prob is >= p1
    threshold_pairs = [(bigfloat.BigFloat(0.0), cumulative_prob)]
    while float(cumulative_prob) > 0:
        p_A = probs[idx_A]
        p_B = probs[idx_B]
        if p_A == p_B:
            cumulative_prob -= 2.0 * p_A
            threshold_pairs.append((p_A, cumulative_prob))
            idx_A += inc_A
            idx_B += inc_B
        elif p_A < p_B:
            cumulative_prob -= p_A
            threshold_pairs.append((p_A, cumulative_prob))
            idx_A += inc_A
        else:
            assert p_B < p_A
            cumulative_prob -= p_B
            threshold_pairs.append((p_B, cumulative_prob))
            idx_B += inc_B

    __precomputed_cumulative_probs__[(int(S), p)] = threshold_pairs

# p -- coin prob
# t -- threshold value
# S -- total number of flips
def __get_prob_that_specific_binomial_prob_over_threshold__(p, t, S):
    if (int(S), p) not in __precomputed_cumulative_probs__:
        __precompute_cumulative_probs_for_binomial__(p, S)

    cum_probs = __precomputed_cumulative_probs__[(int(S), p)]

    # Find the largest prob <= t
    criterion = (lambda x: (lambda y: y[0] <= x))(t)

    # Get the meta-prob that prob is >= t
    (prob, meta_prob) = \
        binary_search_for_last_meeting_criterion(cum_probs, criterion)

    return meta_prob

__memoized_worst_meta_probs_for_thresholds__ = {}

# t -- threshold_value
# S -- total number of flips
def __get_worst_meta_binomial_bound_for_inner_threshold__(t, S, \
        probs_to_check=20000, \
        p_min=bigfloat.exp2(-1000), p_max=bigfloat.BigFloat(0.5)):

    key = (p_min, p_max, probs_to_check, t, S)
    if key in __memoized_worst_meta_probs_for_thresholds__:
        return __memoized_worst_meta_probs_for_thresholds__[key]

    log_p_min=bigfloat.log2(p_min)
    log_p_max=bigfloat.log2(p_max)

    log_increment = (log_p_max - log_p_min) / (probs_to_check - 1)

    worst_value = bigfloat.BigFloat(1.0)

    for i in range(0, probs_to_check):
        curr_log_p = log_p_min + (i * log_increment)
        curr_p = bigfloat.exp2(curr_log_p)

        value = __get_prob_that_specific_binomial_prob_over_threshold__(\
            curr_p, t, S)

        if value < worst_value:
            worst_value = value

    __memoized_worst_meta_probs_for_thresholds__[key] = worst_value
    return worst_value

__memoized_master_values__ = {}

def best_binomial_bound_for_binomial(C, p, P_N, S, num_thresholds=20000):

    key = (C, p, P_N, S, num_thresholds)
    if key in __memoized_master_values__:
        return __memoized_master_values__[key]

    P_X_given_N = bigfloat_prob_of_count_given_p(C, p, S)

    min_thresh = bigfloat.exp2(-1000)
    # Max thresh is the peak of the 50-50 binomial. Beyond that there can be no
    #   guarantees, so it's not worth checking.
    max_thresh = \
        bigfloat_prob_of_count_given_p(int(S / 2), bigfloat.BigFloat(0.5), S)

    log_min_thresh = bigfloat.log2(min_thresh)
    log_max_thresh = bigfloat.log2(max_thresh)
    log_increment = (log_max_thresh - log_min_thresh) / (num_thresholds - 1)

    P_not_N = 1.0 - P_N
    PN_Combo = P_X_given_N * P_N

    best_value = bigfloat.BigFloat(0.0)
    for i in range(0, num_thresholds):
        log_thresh = log_min_thresh + i * log_increment
        thresh = bigfloat.exp2(log_thresh)

        outer_bound_for_thresh = \
            __get_worst_meta_binomial_bound_for_inner_threshold__(thresh, S)

        possible_p_star = outer_bound_for_thresh * \
                              (PN_Combo / (PN_Combo + thresh * P_not_N)) + \
                          (1.0 - outer_bound_for_thresh) * P_N
        value = (1.0 - possible_p_star) * P_not_N / (possible_p_star * P_N)

        if value > best_value:
            best_value = value

    __memoized_master_values__[key] = best_value
    return best_value


"""
# b -- a threshold
# p -- the probability of the coin
# S -- the total number of coin flips
#
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
"""


if __name__ == "__main__":
    best_binomial_bound_for_binomial(C=133, p=bigfloat.BigFloat(0.5), P_N=bigfloat.BigFloat(0.5), S=600, num_thresholds=20000)
    exit(0)


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
