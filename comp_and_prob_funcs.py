import numpy as np
import math
import bigfloat

def bigfloat_fast_choose(A, B):
    A = bigfloat.BigFloat(A)
    B = bigfloat.BigFloat(B)
    if B == 0.0 or B == A:
        return bigfloat.BigFloat(1.0)
    if B == 1.0 or B == (A - 1.0):
        return A

    pi = bigfloat.const_pi()
    first_part = bigfloat.sqrt(A / (2.0 * pi * B * (A - B)))
    second_part = bigfloat_fast_exact_pow(A / B, B)
    third_part = bigfloat_fast_exact_pow(A / (A - B), A - B)
    return first_part * second_part * third_part

def bigfloat_fast_choose_large(A, B):
    A = bigfloat.BigFloat(A)
    B = bigfloat.BigFloat(B)
    if B == 0.0 or B == A:
        return bigfloat.BigFloat(1.0)

    pi = bigfloat.const_pi()
    e = bigfloat.exp(1.0)
    first_part = bigfloat.sqrt(A / (B * (A - B))) * (e / (2.0 * pi))
    second_part = bigfloat_fast_exact_pow(A / B, B)
    third_part = bigfloat_fast_exact_pow(A / (A - B), A - B)
    return first_part * second_part * third_part

def bigfloat_fast_choose_small(A, B):
    A = bigfloat.BigFloat(A)
    B = bigfloat.BigFloat(B)
    if B == 0.0 or B == A:
        return bigfloat.BigFloat(1.0)

    pi = bigfloat.const_pi()
    e = bigfloat.exp(1.0)
    first_part = bigfloat.sqrt(2.0 * pi * A / (B * (A - B))) / (e * e)
    second_part = bigfloat_fast_exact_pow(A / B, B)
    third_part = bigfloat_fast_exact_pow(A / (A - B), A - B)
    return first_part * second_part * third_part

# Uses Stirling's Approximation
def bigfloat_fast_factorial_large(X):
    e = bigfloat.exp(1.0)
    return e * bigfloat_fast_exact_pow(X, X + 0.5) * bigfloat_fast_exact_pow(e, -X)

# Uses Stirling's Approximation
def bigfloat_fast_factorial_small(X):
    e = bigfloat.exp(1.0)
    pi = bigfloat.const_pi()
    result = bigfloat.sqrt(2.0 * pi) * bigfloat_fast_exact_pow(X, X + 0.5) * bigfloat_fast_exact_pow(e, -X)
    return result

def bigfloat_fast_exact_pow(X, Y):
    if Y == 0.0:
        return bigfloat.BigFloat(1.0)
    sign = 1.0 - 2.0 * float(int(Y < 0.0))
    Y = bigfloat.abs(Y)
    base = bigfloat.floor(Y)
    extra = Y - base
    addendum = bigfloat.pow(X, sign * extra)
    if base == 0.0:
        return addendum
    exps = [bigfloat.BigFloat(1)]
    vals = [bigfloat.pow(X, sign)]
    while exps[-1] < base:
        exps.append(2 * exps[-1])
        vals.append(vals[-1] * vals[-1])
    total_result = addendum
    total_exp = bigfloat.BigFloat(0.0)
    for i in range(0, len(exps)):
        idx = len(exps) - (i + 1)
        exp = exps[idx]
        if total_exp + exp <= base:
            total_exp += exp
            total_result = total_result * vals[idx]
        if total_exp == base:
            break
    return total_result

def bigfloat_choose(A, B):
    if A == int(A) and B == int(B):
        A_choose_B = bigfloat.BigFloat(1.0)

        B = min(B, A - B)
        for i in range(0, int(B)):
            A_choose_B *= (A - i)
            A_choose_B /= (i + 1)

        return A_choose_B

    top_gamma = bigfloat.lngamma(A + 1.0)
    bot_gamma = bigfloat.lngamma(B + 1.0) + bigfloat.lngamma((A - B) + 1.0)
    return bigfloat.exp(top_gamma - bot_gamma)

def bigfloat_prob_of_count_given_p(C, p, S):
    assert float(p) <= 1.0
    assert float(C) <= float(S)

    zero = bigfloat.BigFloat(0.0)
    one = bigfloat.BigFloat(1.0)

    # Handle p == 0 and p == 1 with special cases due to pow issues.
    if p == zero:
        if int(C) == 0:
            return one
        else:
            return zero
    elif p == one:
        if int(C) == int(S):
            return one
        else:
            return zero

    C = bigfloat.BigFloat(C)
    S = bigfloat.BigFloat(S)
    p = bigfloat.BigFloat(p)

    # print("    Computing %f^%d..." % (p, C))
    prob = bigfloat.pow(p, C)
    # Check to see if the bigfloat ran out of resolution:
    if zero == prob:
        print("Not enough bigfloat bits for pow(%f, %d). Using slow method..." % (p, C))
        return bigfloat_slow_prob_of_count_given_p(C, p, S)

    # print("    Computing %d choose %d..." % (S, C))
    prob *= bigfloat_choose(S, C)
    # Check to see if the bigfloat ran out of resolution:
    if bigfloat.is_inf(prob):
        print("Not enough bigfloat bits for %d choose %d. Using slow method..." % (S, C))
        print(prob.precision)
        return bigfloat_slow_prob_of_count_given_p(C, p, S)

    # print("    Computing %f^%d" % (1.0 - p, S - C))
    prob *= bigfloat.pow(1.0 - p, S - C)
    # Check to see if the bigfloat ran out of resolution:
    if zero == prob:
        print("Not enough bigfloat bits for pow(1.0 - %f, %d). Using slow method..." % (1.0 - p, S - C))
        return bigfloat_slow_prob_of_count_given_p(C, p, S)

    if float(prob) > 1.0:
        print("Error! Got prob > 1.0 from params C = %f, p = %f, S = %f" % (C, p, S))
    assert 0.0 <= float(prob) and float(prob) <= 1.0
    assert float(prob) != float('inf') and float(prob) != float('-inf') and float(prob) != float('nan')

    return prob

def bigfloat_fast_prob_of_count_given_p(C, p, S):
    assert 0.0 <= p
    assert p <= 1.0
    assert float(C) <= float(S)

    zero = bigfloat.BigFloat(0.0)
    one = bigfloat.BigFloat(1.0)

    # Handle p == 0 and p == 1 with special cases due to pow issues.
    if p == zero:
        if C == 0:
            return one
        else:
            return zero
    elif p == one:
        if C == S:
            return one
        else:
            return zero

    C = bigfloat.BigFloat(C)
    S = bigfloat.BigFloat(S)
    p = bigfloat.BigFloat(p)

    # print("    Estimating %f^%d..." % (p, C))
    prob = bigfloat_fast_exact_pow(p, C)
    # Check to see if the bigfloat ran out of resolution:
    if zero == prob:
        print("Not enough bigfloat bits for pow(%f, %d). Using slow method..." % (p, C))
        return bigfloat_slow_prob_of_count_given_p(C, p, S)

    # print("    Estimating %d choose %d..." % (S, C))
    bf_fc = bigfloat_fast_choose_large(S, C)
    assert bf_fc > 0
    prob *= bf_fc
    # Check to see if the bigfloat ran out of resolution:
    if bigfloat.is_inf(prob):
        print("Not enough bigfloat bits for %d choose %d. Using slow method..." % (S, C))
        print(prob.precision)
        return bigfloat_slow_prob_of_count_given_p(C, p, S)

    # print("    Estimating %f^%d..." % (1.0 - p, S - C))
    prob *= bigfloat_fast_exact_pow(1.0 - p, S - C)
    # Check to see if the bigfloat ran out of resolution:
    if zero == prob:
        print("Not enough bigfloat bits for pow(1.0 - %f, %d). Using slow method..." % (1.0 - p, S - C))
        return bigfloat_slow_prob_of_count_given_p(C, p, S)

    if float(prob) > 1.0:
        print("Error! Got prob > 1.0 from params C = %f, p = %f, S = %f" % (C, p, S))
    assert 0.0 <= float(prob)
    assert float(prob) <= 1.1  # NOTE! 1.1 rather than 1.0 to allow for some "slop" in the probs.
    assert float(prob) != float('inf') and float(prob) != float('-inf') and float(prob) != float('nan')

    return prob

def bigfloat_slow_prob_of_count_given_p(C, p, S):
    assert float(p) <= 1.0
    assert float(C) <= float(S)

    C = bigfloat.BigFloat(C)
    S = bigfloat.BigFloat(S)
    p = bigfloat.BigFloat(p)
    p_not = 1.0 - p

    C_min = bigfloat.min(C, S - C)
    under_one_vals = [p for i in range(0, int(C))] + \
                     [p_not for i in range(0, int(S) - int(C))]
    over_one_vals = [(S - i) / (C_min - (i + 1)) for i in range(0, int(C_min))]

    result = bigfloat.BigFloat(1.0)

    while len(over_one_vals) > 0 or len(under_one_vals) > 0:
        if len(over_one_vals) == 0:
            result *= under_one_vals.pop()
            continue
        if len(under_one_vals) == 0:
            result *= over_one_vals.pop()
            continue
        a = over_one_vals[-1]
        b = under_one_vals[-1]
        v1 = float(result) * a
        v2 = float(result) * b
        assert a > 0.0
        assert b > 0.0
        a_diff = abs(1.0 - v1)
        b_diff = abs(1.0 - v1)
        if a_diff < b_diff:
            result *= over_one_vals.pop()
        else:
            result *= under_one_vals.pop()

    if float(result) > 1.0:
        print("Error! Got prob > 1.0 from params C = %f, p = %f, S = %f" % (C, p, S))
        assert float(result) <= 1.0

    return result
