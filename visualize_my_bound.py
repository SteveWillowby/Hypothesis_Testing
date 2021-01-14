import bigfloat
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from my_bound_funcs import my_universal_bound_on_binomial


# num_flips -- the total number of flips of the coin (not the number of heads)
# coin_prob -- the coin's probability
# marked_points -- a list of (C, P_N) points to highlight on the 3d contour
# num_priors -- the number of distinct prior values for which to compute the surface
def visualize_my_universal_bound_on_binomial(num_flips, coin_prob=0.5, marked_points=[], num_priors=1001):
    start_P_N = bigfloat.pow(2.0, -30)
    end_P_N = 1.0 - start_P_N
    P_N_increment = (end_P_N - start_P_N) / (num_priors - 1)
    P_N = start_P_N
    
    start_C = 0
    end_C = num_flips
    C_increment = 1
    C = start_C

    evidence_vals = []
    C_vals = []


    while C <= end_C:
        C_vals.append(float(C))
        evidence_vals.append([])

        P_N_vals = []
        P_N = start_P_N
        while P_N <= end_P_N:
            P_N_vals.append(float(P_N))

            evidence_bound = my_universal_bound_on_binomial(num_flips, C, coin_prob, P_N)
            evidence_vals[-1].append(float(bigfloat.log2(evidence_bound)))

            P_N += P_N_increment
        C += C_increment

    # Plotting Data

    # Have the "major" axis be C_vals and the "secondary" axis be P_N_vals
    P_N_vals_2d = np.array([P_N_vals for _ in C_vals])
    C_vals_2d = np.array([[C_vals[i] for _ in P_N_vals] for i in range(0, len(C_vals))])

    evidence_vals = np.array(evidence_vals)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # plt.hold(True)  # Necessary to have both the scatter points and the surface.

    marked_points = [(float(a), float(b)) for (a, b) in marked_points]
    marked_points = [(a, b, float(my_universal_bound_on_binomial(num_flips, a, coin_prob, b))) \
        for (a, b) in marked_points]
    marked_x = [a for (a, _, _) in marked_points]
    marked_y = [b for (_, b, _) in marked_points]
    marked_z = [c for (_, _, c) in marked_points]
    points_plot = ax.scatter(marked_x, marked_y, marked_z, c="green", alpha=1)

    surface_plot = ax.plot_surface(C_vals_2d, P_N_vals_2d, evidence_vals, \
               linewidth=0, antialiased=False, cmap=plt.get_cmap('RdBu_r'), alpha=0.5)  # 'Blues'

    plt.suptitle("Log_2 of Strength of Evidence Against Null (N)")
    plt.xlabel("Number of Heads H")
    plt.ylabel("Prior Value, P(N)")
    plt.title("Log_2( P(H | not-N) / P(H | N) )")

    plt.show()

if __name__ == "__main__":
    bf_context = bigfloat.Context(precision=2000, emax=100000, emin=-100000)
    bigfloat.setcontext(bf_context)

    visualize_my_universal_bound_on_binomial(600, \
        coin_prob=(1.0 / bigfloat.BigFloat(6.0)), \
        marked_points=[(113, 0.5)], num_priors=101)
