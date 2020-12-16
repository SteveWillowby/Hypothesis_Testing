import bigfloat
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pvalue_funcs import pvalue_for_binomial, precalc_pvalue_for_binomial
from bayes_factor_funcs import bayes_factor_for_binomial
from my_bound_funcs import my_bound_for_binomial

if __name__ == "__main__":
    # Parameters for plots

    bf_context = bigfloat.Context(precision=2000, emax=100000, emin=-100000)
    bigfloat.setcontext(bf_context)
    P_N = bigfloat.BigFloat(0.5)

    start_S = bigfloat.BigFloat(10)
    end_S = bigfloat.BigFloat(100)
    S = start_S

    tests_per_S = 100

    pvalue_cutoff = 0.05

    S_values = []
    S_v = 1.0
    
    pvalue_correct_trues = []
    pvalue_correct_falses = []
    bayes_factor_correct_trues = []
    bayes_factor_correct_falses = []
    my_bound_correct_trues = []
    my_bound_correct_falses = []

    while S <= end_S:
        pct = 0
        pcf = 0
        bct = 0
        bcf = 0
        mct = 0
        mcf = 0
        for _ in range(0, tests_per_S):
            null_C = bigfloat.BigFloat(np.random.binomial(S, 0.5))
            if precalc_pvalue_for_binomial(S, null_C, 0.5) >= pvalue_cutoff:
                pct += 1
            bf = bayes_factor_for_binomial(S, null_C, 0.5)
            if bf <= 1.0:
                bct += 1
            if my_bound_for_binomial(S, null_C, 0.5) <= 1.0:  #TODO: Update
                mct += 1

            alt_coin_prob = np.random.uniform(0.0, 1.0)
            alt_C = bigfloat.BigFloat(np.random.binomial(S, alt_coin_prob))
            if precalc_pvalue_for_binomial(S, alt_C, 0.5) < pvalue_cutoff:
                pcf += 1
            if bayes_factor_for_binomial(S, alt_C, 0.5) > 1.0:
                bcf += 1
            if my_bound_for_binomial(S, alt_C, 0.5) > 1.0:
                mcf += 1

        print("------------")
        print("pct: %d" % pct)
        print("pcf: %d" % pcf)
        print("bct: %d" % bct)
        print("bcf: %d" % bcf)
        print("mct: %d" % mct)
        print("mcf: %d" % mcf)

        S_values.append(S_v)
        pvalue_correct_trues.append(pct / float(tests_per_S))
        pvalue_correct_falses.append(pcf / float(tests_per_S))
        bayes_factor_correct_trues.append(bct / float(tests_per_S))
        bayes_factor_correct_falses.append(bcf / float(tests_per_S))
        my_bound_correct_trues.append(mct / float(tests_per_S))
        my_bound_correct_falses.append(mcf / float(tests_per_S))

        S *= 10.0
        S_v += 1.0

    plt.scatter(S_values, pvalue_correct_trues)
    plt.scatter(S_values, bayes_factor_correct_trues)
    plt.scatter(S_values, my_bound_correct_trues)
    plt.title("Fraction of Test Results Wherein Method Correctly Indicates _no_ Evidence Against Null")
    plt.xlabel("log_10(S)")
    plt.ylabel("Fraction of Test Results")
    plt.show()

    plt.scatter(S_values, pvalue_correct_falses)
    plt.scatter(S_values, bayes_factor_correct_falses)
    plt.scatter(S_values, my_bound_correct_falses)
    plt.title("Fraction of Test Results Wherein Method Correctly Indicates Evidence Against Null")
    plt.xlabel("log_10(S)")
    plt.ylabel("Fraction of Test Results")
    plt.show()


    """
    start_P_N = bigfloat.pow(2.0, -20)
    end_P_N = bigfloat.BigFloat(1.0) - start_P_N
    # start_P_N = bigfloat.BigFloat(1.0) / 3.0
    # end_P_N = bigfloat.BigFloat(2.0) / 3.0
    P_N_increment = (end_P_N - start_P_N) / 10.0

    coin_prob = 0.5

    S = 10

    start_C = 0
    end_C = S
    C_increment = 1

    evidence_vals = []
    C = start_C
    C_vals = []
    while C <= end_C:
        C_vals.append(float(C))
        evidence_vals.append([])

        P_N_vals = []
        P_N = start_P_N
        while P_N <= end_P_N:
            P_N_vals.append(float(P_N))

            print("C: %d -- P_N: %f" % (C, P_N))
            (best_m, best_evidence_strength) = best_m_and_evidence_strength(C, coin_prob, S + 1, P_N)
            if best_m is None:
                best_m = "N/A"
            else:
                best_m = str(float(best_m))
            print("    Best m: %s -- Best evidence strength: %f" % (best_m, best_evidence_strength))
            evidence_vals[-1].append(float(best_evidence_strength))

            P_N += P_N_increment
        C += C_increment

    # Plotting Data

    # Have the "major" axis be C_vals and the "secondary" axis be P_N_vals
    P_N_vals_2d = np.array([P_N_vals for _ in C_vals])
    C_vals_2d = np.array([[C_vals[i] for _ in P_N_vals] for i in range(0, len(C_vals))])

    evidence_vals = np.array(evidence_vals)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(C_vals_2d, P_N_vals_2d, evidence_vals, cmap=cm.coolwarm,
               linewidth=0, antialiased=False)

    plt.suptitle("Log of Strength of Evidence Against Null (N)")
    plt.xlabel("Number of Heads H")
    plt.ylabel("Prior Value, P(N)")
    plt.title("P(N | not-H) / P(N | H)")

    plt.show()

    """
