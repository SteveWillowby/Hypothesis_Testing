import bigfloat
from plotting_funcs import plot_3d_surface
from my_bound_funcs import my_universal_bound_on_binomial, my_universal_bound


# num_flips -- the total number of flips of the coin (not the number of heads)
# coin_prob -- the coin's probability
# marked_points -- a list of (C, P_N) points to highlight on the 3d contour
# num_priors -- the number of distinct prior values for which to compute the surface
def visualize_my_universal_bound_on_binomial(num_flips, coin_prob=0.5, \
        marked_points=[], num_priors=101):
    end_P_N = bigfloat.pow(2.0, -20)
    start_P_N = 1.0 - end_P_N
    P_N_increment = (end_P_N - start_P_N) / (num_priors - 1)
    P_N_vals = [start_P_N + (i * P_N_increment) for i in range(0, num_priors)]
    
    start_C = 0
    C_increment = 1
    C_vals = [start_C + (i * C_increment) for i in range(0, num_flips + 1)]

    z_axis_function = (lambda x: lambda y: \
        float(bigfloat.log2(my_universal_bound_on_binomial(\
            x[0], y[0], x[1], y[1]))))((num_flips, coin_prob))

    surface_alpha=0.5
    if len(marked_points) == 0:
        surface_alpha = 0.8

    plot_3d_surface(C_vals, P_N_vals, z_axis_function, \
        surface_alpha=surface_alpha, marked_points=marked_points, \
        plot_title="Log_2 of Strength of Evidence Against Null, N", \
        left_axis_name="Number of Sixes, X", \
        right_axis_name="Prior Value, P(N)", \
        plot_subtitle="Log_2( P(X | not-N) / P(X | N) )")
    return

# n -- the size of the experiment's result space
# num_probs -- the number of distinct result probs P(X | N) for which to compute the surface
# num_priors -- the number of distinct prior values P(N) for which to compute the surface
# marked_points -- a list of (P_X_given_N, P_N) points to highlight on the 3d contour
def visualize_my_universal_bound_on_plain_probs(n, num_probs=101, num_priors=101, marked_points=[]):
    end_P_N = bigfloat.pow(2.0, -20)
    start_P_N = 1.0 - end_P_N
    P_N_increment = (end_P_N - start_P_N) / (num_priors - 1)
    P_N_vals = [start_P_N + (i * P_N_increment) for i in range(0, num_priors)]
    
    start_P_X_given_N = start_P_N
    end_P_X_given_N = end_P_N
    P_X_given_N_increment = (end_P_X_given_N - start_P_X_given_N) / (num_probs - 1)
    P_X_given_N_vals = [start_P_X_given_N + (i * P_X_given_N_increment) for \
        i in range(0, num_probs)]

    z_axis_function = (lambda x: lambda y: \
        float(bigfloat.log2(my_universal_bound(y[0], y[1], x))))(n)

    surface_alpha=0.5
    if len(marked_points) == 0:
        surface_alpha = 0.8

    plot_3d_surface(P_X_given_N_vals, P_N_vals, z_axis_function, \
        surface_alpha=surface_alpha, marked_points=marked_points, \
        plot_title="Strength of Evidence Against Null (N) when n = %d" % n, \
        left_axis_name="Probability of Result According to Null, P(X | N)", \
        right_axis_name="Prior Value, P(N)", \
        plot_subtitle="")

# n -- the size of the experiment's result space
# num_probs -- the number of distinct result probs P(X | N) for which to compute the surface
# num_priors -- the number of distinct prior values P(N) for which to compute the surface
# marked_points -- a list of (P_X_given_N, P_N) points to highlight on the 3d contour
def visualize_my_universal_bound_on_plain_probs_evidence_only(n, num_probs=101, num_priors=101, marked_points=[]):
    end_P_N = bigfloat.pow(2.0, -100)
    start_P_N = 1.0 - end_P_N

    end_P_N = bigfloat.log(end_P_N)
    start_P_N = bigfloat.log(start_P_N)

    P_N_increment = (end_P_N - start_P_N) / (num_priors - 1)
    P_N_vals = [start_P_N + (i * P_N_increment) for i in range(0, num_priors)]
    
    start_P_X_given_N = 1.0 / bigfloat.BigFloat(n)
    end_P_X_given_N = end_P_N

    start_P_X_given_N = bigfloat.log(start_P_X_given_N)

    P_X_given_N_increment = (end_P_X_given_N - start_P_X_given_N) / (num_probs - 1)
    P_X_given_N_vals = [start_P_X_given_N + (i * P_X_given_N_increment) for \
        i in range(0, num_probs)]

    z_axis_function = (lambda x: lambda y: \
        float(bigfloat.log2(my_universal_bound(bigfloat.exp(y[0]), bigfloat.exp(y[1]), x))))(n)

    surface_alpha=0.5
    if len(marked_points) == 0:
        surface_alpha = 0.8

    plot_3d_surface(P_X_given_N_vals, P_N_vals, z_axis_function, \
        surface_alpha=surface_alpha, marked_points=marked_points, \
        plot_title="Log of Strength of Evidence Against Null (N) when n = %d" % n, \
        left_axis_name="log(P(X | N))", \
        right_axis_name="log(P(N))", \
        plot_subtitle="")

if __name__ == "__main__":
    bf_context = bigfloat.Context(precision=2000, emax=100000, emin=-100000)
    bigfloat.setcontext(bf_context)

    """
    visualize_my_universal_bound_on_binomial(60, \
        coin_prob=(1.0 / bigfloat.BigFloat(6.0)), \
        marked_points=[(13, 0.5)], num_priors=11)

    """
    visualize_my_universal_bound_on_plain_probs_evidence_only(10, \
        num_probs=101, num_priors=101, marked_points=[])
