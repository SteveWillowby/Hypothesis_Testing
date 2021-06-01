from comb_and_prob_funcs import *
from higher_order_basics import *
from plotting_funcs import *
import random
import numpy as np
import optimizing
import bigfloat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Can pass "TV", "H", "JS", or "L2" for metric
def test_for_higher_order_convergence_with_binomials(null_p=0.5, \
        coin_tosses=50, heads=[20, 25], \
        num_dists_by_order=[10000, 5000, 2500, 1250], \
        order_names=["First", "Second", "Third", "Fourth"], \
        metric="TV"):

    axis_fontsize = 14
    legend_fontsize = 12
    title_fontsize = 14

    if metric == "TV":
        dm = total_variation_distance
    elif metric == "H":
        dm = hellinger_distance
    elif metric == "L2":
        dm = L2_distance
    elif metric == "JS":
        dm = jensen_shannon_distance
    else:
        print("Unrecognized metric name '%s'!!!!!" % metric)

    binomial = (lambda n : (lambda p_list : binomial_dist(n, p_list[0])))(coin_tosses)

    print("Creating Null Dist")
    null_dist = binomial_dist(coin_tosses, null_p)
    print("  Null Dist Complete")

    # If ensure_null_present is true, then the sample is biased and not QUITE
    #   representative.
    ensure_null_present = False

    print("Building Uniform Dist over First Order Dists")
    (first_order_credal_set_sample_space, uniform_measure) = \
        uniform_dist_over_parametrized_credal_set(\
            param_intervals=[[bigfloat.BigFloat(0.0), bigfloat.BigFloat(1.0)]],\
            params_to_dist=binomial, \
            distance_metric=dm, \
            num_points_per_param=(num_dists_by_order[0] / 10 + 1))


    U_TV_2 = np.matmul(np.array([uniform_measure]), \
                       first_order_credal_set_sample_space)
    U_TV_2 = U_TV_2[0]
    print("  Done Building Uniform Dist over First Order Dists")

    space_size = len(uniform_measure)
    x_axis = [bigfloat.BigFloat(1.0) / (space_size - 1) * i \
                for i in range(0, space_size)]
    y_axis = uniform_measure * (len(uniform_measure) - 1)
    plt.plot(x_axis, y_axis)
    plt.title("%s-Uniform PDF Over Proportion p for n = %d" % (metric, coin_tosses), fontsize=title_fontsize)
    plt.xlabel("Proportion p", fontsize=axis_fontsize)
    plt.ylabel("Probability Density", fontsize=axis_fontsize)
    plt.tight_layout()  # Makes sure the axis labels are allowed on the plot.
    figure_name = "figures/binomial_convergence/%s_uniform_over_parameter" % metric
    save_figure_with_data_csv(plt, figure_name, [x_axis], [y_axis])
    # plt.show()
    plt.close()

    done = False
    while not done:
        print("Creating Representative First Order Sample via Uniform Measure")

        first_order_dists = samples_from_discrete_dist(\
            elements=first_order_credal_set_sample_space, \
            dist=uniform_measure, \
            num_samples=num_dists_by_order[0])

        if ensure_null_present:
            for dist in first_order_dists:
                if dist.all() == null_dist.all():
                    done = True
                    break
            if not done:
                print("Issue! Didn't get null dist in 1st order dists - retrying.")
        else:
            done = True
    print("  Sample Complete")

    print("Generating Uniform Dist over First Order Representative Sample")
    uniform_second_order_dist = np.array(first_order_dists[0])
    for i in range(1, len(first_order_dists)):
        uniform_second_order_dist += first_order_dists[i]
    uniform_second_order_dist /= np.sum(uniform_second_order_dist)
    print("  Generating Uniform Dist Complete")

    print("Plotting Uniform Second Order Dist(s)")
    x_axis = [i for i in range(0, coin_tosses + 1)]
    y_axis = uniform_second_order_dist
    plt.plot(x_axis, y_axis)
    plt.title("Distribution Implied by 2nd Order %s-Uniform" % metric, fontsize=title_fontsize)
    plt.xlabel("Number of Heads", fontsize=axis_fontsize)
    plt.ylabel("Probability", fontsize=axis_fontsize)
    plt.tight_layout()  # Makes sure the axis labels are allowed on the plot.
    figure_name = "figures/binomial_convergence/%s_sampled_second_order_uniform_over_heads" % metric
    save_figure_with_data_csv(plt, figure_name, [x_axis], [y_axis])
    # plt.show()
    plt.close()

    x_axis = [i for i in range(0, coin_tosses + 1)]
    y_axis = U_TV_2
    plt.plot(x_axis, y_axis)
    plt.title("Distribution Implied by 2nd Order %s-Uniform" % metric, fontsize=title_fontsize)
    plt.xlabel("Number of Heads", fontsize=axis_fontsize)
    plt.ylabel("Probability", fontsize=axis_fontsize)
    plt.tight_layout()  # Makes sure the axis labels are allowed on the plot.
    figure_name = "figures/binomial_convergence/%s_second_order_uniform_over_heads" % metric
    save_figure_with_data_csv(plt, figure_name, [x_axis], [y_axis])
    # plt.show()
    plt.close()
    print("  Plotting of Uniform Second Order Dist(s) Complete")

    print("Getting Chances of %s Heads from %d Tosses" % (heads, coin_tosses))
    first_order_chances = [sorted([dist[heads[i]] for \
                                dist in first_order_dists]) for \
                                    i in range(0, len(heads))]
    print("  Getting Chances Complete")

    orders_chances = [first_order_chances]

    new_dists = np.array(first_order_dists)
    for order_idx in range(1, len(num_dists_by_order)):
        # old_dists_transposed = new_dists.transpose()
        num_dists = num_dists_by_order[order_idx]
        order_name = order_names[order_idx]
        # new_dists = []
        print("Working on %s Order Prob Functions" % order_name)
        new_dists = generate_n_random_dist_over_dists(num_dists, new_dists, metric, use_128_bits=True)
        # for i in range(0, num_dists):
        #     new_dists.append(generate_random_dist_over_dists(\
        #                         old_dists_transposed, \
        #                         metric))
        # new_dists = np.array(new_dists)
        print("  Accumulated %s Order Prob Functions" % order_name)

        order_chances = [sorted([dist[heads[i]] for dist in new_dists]) for \
                            i in range(0, len(heads))]
        orders_chances.append(order_chances)

    for heads_idx in range(0, len(heads)):
        for start_order in range(0, len(orders_chances)):

            x_axes = []
            y_axes = []

            heads_num = heads[heads_idx]
            print("Plotting Ordered Chances of %d heads from %d Tosses" % \
                    (heads_num, coin_tosses))

            centerpoint = np.float128(uniform_second_order_dist[heads_num])
            plot_min = min(\
                np.float128(orders_chances[start_order][heads_idx][0]), \
                centerpoint)
            plot_max = max(\
                np.float128(orders_chances[start_order][heads_idx][-1]), \
                centerpoint)

            height = plot_max - plot_min
            plot_min -= 0.05 * height
            plot_max += 0.05 * height

            for i in range(0, len(orders_chances)):
                order_chances = orders_chances[i][heads_idx]
                x_axis = [bigfloat.BigFloat(j) / (len(order_chances) - 1) \
                    for j in range(0, len(order_chances))]
                x_axes.append(x_axis)
                y_axes.append(order_chances)
                plt.plot(x_axis, order_chances, label=("%s Order Distributions" % order_names[i]))

            x_axis = [0, 1]
            y_axis = [uniform_second_order_dist[heads_num], uniform_second_order_dist[heads_num]]
            x_axes.append(x_axis)
            y_axes.append(y_axis)
            plt.plot(x_axis, y_axis, linestyle="dashed", label="Second Order Uniform")

            plt.ylim((plot_min, plot_max))

            title = "Convergence for Event: [%d Heads on %d Tosses]" % \
                (heads_num, coin_tosses)
            """
            suptitle = "Chances of Event: [%d Heads on %d Tosses]" % \
                (heads_num, coin_tosses)
            title = "For"
            for i in range(start_order, len(order_names) - 1):
                title += " %s," % order_names[i]
            if start_order < len(order_names) - 1:
                title += " and %s Order Confidences" % order_names[-1]
            else:
                title += " %s Order Confidences" % order_names[-1]
            """
            if start_order == 0:
                # plt.suptitle(suptitle, fontsize=title_fontsize)
                plt.title(title, fontsize=title_fontsize)
            if start_order == 2:
                plt.xlabel("Probability Functions", fontsize=axis_fontsize)
            if heads_idx == 0:
                plt.ylabel("Chance of Event", fontsize=axis_fontsize)
            plt.legend(fontsize=legend_fontsize)
            plt.tight_layout()  # Makes sure the axis labels are allowed on the plot.
            figure_name = "figures/binomial_convergence/%s_higher_order_convergence_%d_%d_%d" % \
                                (metric, heads_num, coin_tosses, start_order + 1)
            save_figure_with_data_csv(plt, figure_name, x_axes, y_axes)
            # plt.show()
            plt.close()

            print("  Plotting Ordered Chances Complete")

# n - number of tosses
# c_values_to_plot - a list of #heads to show the likelihood values for
def binomial_likelihood_function_plot(n, c_values_to_plot, num_binoms=10000, metric="TV"):
    if metric == "TV":
        dm = total_variation_distance
    elif metric == "H":
        dm = hellinger_distance
    elif metric == "L2":
        dm = L2_distance
    elif metric == "JS":
        dm = jensen_shannon_distance
    else:
        print("Unrecognized metric name '%s'!!!!!" % metric)

    binomial = (lambda t : (lambda p_list : binomial_dist(t, p_list[0])))(n)

    (binomials, uniform_measure) = \
        uniform_dist_over_parametrized_credal_set(\
            param_intervals=[[bigfloat.BigFloat(0.0), bigfloat.BigFloat(1.0)]],\
            params_to_dist=binomial, \
            distance_metric=dm, \
            num_points_per_param=num_binoms)

    U_TV_2 = np.matmul(np.array([uniform_measure]), binomials)
    U_TV_2 = U_TV_2[0]

    proportions = [bigfloat.BigFloat(j) / (num_binoms - 1) \
                        for j in range(0, num_binoms)]

    axis_fontsize = 14
    legend_fontsize = 12
    title_fontsize = 14

    for c in c_values_to_plot:
        ref_value = U_TV_2[c]
        hocs_ratios = [b[c] / ref_value for b in binomials]
        plt.plot(proportions, hocs_ratios, color='black', label="HOCS Ratio")
        plt.plot([0, 1], [1, 1], color='red', linestyle='dashed', label="Evidence Threshold")

        best_ratio_idx = 0
        for i in range(0, len(hocs_ratios)):
            if hocs_ratios[i] > hocs_ratios[best_ratio_idx]:
                best_ratio_idx = i
        plt.plot([proportions[best_ratio_idx], proportions[best_ratio_idx]], \
                 [0, hocs_ratios[best_ratio_idx]], \
                 color='black', linestyle='dashed')

        plt.legend(fontsize=legend_fontsize)
        plt.xlabel("Binomial Proportion p", fontsize=axis_fontsize)
        plt.ylabel("HOCS Ratio", fontsize=axis_fontsize)
        plt.xticks([0.1 * i for i in range(0, 11)])
        plt.suptitle("Evidence For/Against Binomial Proportions", fontsize=title_fontsize)
        plt.title("From %d Heads Out of %d Coin Tosses" % (c, n), fontsize=title_fontsize)
        figure_name = "figures/%s_binomial_%d_%d_hocs_ratios.pdf" % \
                            (metric, c, n)
        save_figure_with_data_csv(plt, figure_name, [proportions], [hocs_ratios])
        plt.show()
        plt.close()

# Consider: Each dist in this order has a difference of just 2.
#   Is this the basis for my desired PDF ordering?
def possible_color_counts_for_n_balls_and_c_colors(n, c):
    if c == 1:
        return [[n]]

    counts = []
    for i in range(0, n):
        basics = possible_color_counts_for_n_balls_and_c_colors(n - i, c - 1)
        for b in basics:
            reverse_b = [b[len(b) - j] for j in range(1, len(b) + 1)]
            if i % 2 == 0:
                counts.append(b + [i])
            else:
                counts.append(reverse_b + [i])
    counts.append([0 for _ in range(1, c)] + [n])
    return counts


def uniform_calcs_for_urn_sampling(num_balls_in_urn, num_colors_possible):
    n = num_balls_in_urn
    c = num_colors_possible
    pcc = possible_color_counts_for_n_balls_and_c_colors(n, c)
    all_dists = [np.array([bigfloat.BigFloat(v) / n for v in d]) for d in pcc]
    all_dists = np.array(all_dists)
    all_dists = all_dists.transpose()

    observed_color = 0
    U_TV_2 = np.array([bigfloat.BigFloat(1.0) / len(all_dists[0]) for \
                            _ in all_dists[0]])

    U_TV_2_over_colors = np.matmul(all_dists, U_TV_2)
    print("Distribution over colors before observing color 0:")
    print(U_TV_2_over_colors)

    U_TV_2_prime = np.array([all_dists[0][i] for i in range(0, len(all_dists[0]))])
    U_TV_2_prime = U_TV_2_prime / np.sum(U_TV_2_prime)

    U_TV_2_prime_over_colors = np.matmul(all_dists, U_TV_2_prime)
    print("Distribution over colors after observing color 0:")
    print(U_TV_2_prime_over_colors)

    # Now for Color 1:
    U_TV_2_prime = np.array([all_dists[1][i] * U_TV_2_prime[i] for i in range(0, len(all_dists[0]))])
    U_TV_2_prime = U_TV_2_prime / U_TV_2_prime_over_colors[1]

    U_TV_2_prime_over_colors = np.matmul(all_dists, U_TV_2_prime)
    print("Distribution over colors after observing color 0 then 1:")
    print(U_TV_2_prime_over_colors)

    

def test_for_a_natural_distance_metric():
    
    # jensen_shannon_distance(dist_A, dist_B)

    # naive_overlap_distance(dist_A, dist_B)

    # hellinger_distance(dist_A, dist_B)

    num_tosses = 100
    num_priors = 101

    start_prior = bigfloat.exp2(-10)
    binomial_dists = \
        np.array(m_binomial_dists_over_n_coin_tosses(m=num_priors, n=num_tosses, \
            start_p=start_prior))

    prior = start_prior
    prior_inc = (1.0 - 2.0 * start_prior) / (num_priors - 1)
    priors = []
    for i in range(0, num_priors):
        priors.append(prior)
        prior += prior_inc

    priors = np.array(priors)

    reference_dists = [binomial_dists[0], binomial_dists[int(num_tosses/3)], binomial_dists[int(num_tosses/2)]]
    js_distances = [[jensen_shannon_distance(d, ref_dist) for d in binomial_dists] for ref_dist in reference_dists]
    no_distances = [[naive_overlap_distance(d, ref_dist) for d in binomial_dists] for ref_dist in reference_dists]
    h_distances = [[hellinger_distance(d, ref_dist) for d in binomial_dists] for ref_dist in reference_dists]
    tv_distances = [[total_variation_distance_for_binomials(d, ref_dist) for d in binomial_dists] for ref_dist in reference_dists]

    plt.plot([i for i in range(0, num_tosses + 1)], reference_dists[0])
    plt.title("Reference Dists")
    plt.xlabel("Number of Heads")
    plt.ylabel("Probability")
    plt.plot([i for i in range(0, num_tosses + 1)], reference_dists[1])
    plt.plot([i for i in range(0, num_tosses + 1)], reference_dists[2])
    plt.show()

    # plt.plot(priors, js_distances[0])
    # plt.plot(priors, no_distances[0])
    # plt.plot(priors, h_distances[0])
    plt.plot(priors, tv_distances[0])
    plt.title("Comparisons of Distances on First Reference")
    plt.xlabel("Proportion Parameter")
    plt.ylabel("Distance from a Reference Distribution")
    plt.show()

    # plt.plot(priors, js_distances[1])
    # plt.plot(priors, no_distances[1])
    # plt.plot(priors, h_distances[1])
    plt.plot(priors, tv_distances[1])
    plt.title("Comparisons of Distances on Second Reference")
    plt.xlabel("Proportion Parameter")
    plt.ylabel("Distance from a Reference Distribution")
    plt.show()

    # plt.plot(priors, js_distances[2])
    # plt.plot(priors, no_distances[2])
    # plt.plot(priors, h_distances[2])
    plt.plot(priors, tv_distances[2])
    plt.title("Comparisons of Distances on Third Reference")
    plt.xlabel("Proportion Parameter")
    plt.ylabel("Distance from a Reference Distribution")
    plt.show()

def test_distance_metrics_for_linearity_of_immediate_space_on_binomials():
    epsilon_exponents = [-32, -64, -128, -512, -1024]
    epsilons = [bigfloat.exp2(v) for v in epsilon_exponents]

    num_tosses = 10
    start_p = bigfloat.BigFloat(0.0)
    end_p = bigfloat.BigFloat(0.5)
    num_p = 1001

    binomials = {}
    values_of_p = []
    for p_idx in range(0, num_p):
        base_p = start_p + ((end_p - start_p) * p_idx) / (num_p - 1)
        values_of_p.append(base_p)
        binomials[base_p] = binomial_dist(num_tosses, base_p)
        for e in epsilons:
            binomials[base_p + e] = binomial_dist(num_tosses, base_p + e)

    js_rates_of_change = []
    no_rates_of_change = []
    l1_rates_of_change = []
    l2_rates_of_change = []
    h_rates_of_change = []
    tv_rates_of_change = []
    for e in epsilons:
        js_roc = []
        no_roc = []
        l1_roc = []
        l2_roc = []
        h_roc = []
        tv_roc = []
        for base_p in values_of_p:
            next_p = base_p + e
            # js_roc.append(jensen_shannon_distance(binomials[base_p], binomials[next_p]) / e)
            no_roc.append(naive_overlap_distance(binomials[base_p], binomials[next_p]) / e)
            l1_roc.append(L1_distance(binomials[base_p], binomials[next_p]) / e)
            l2_roc.append(L2_distance(binomials[base_p], binomials[next_p]) / e)
            # h_roc.append(hellinger_distance(binomials[base_p], binomials[next_p]) / e)
            tv_roc.append(total_variation_distance_for_binomials(binomials[base_p], binomials[next_p]) / e)
        js_rates_of_change.append(js_roc)
        no_rates_of_change.append(no_roc)
        l1_rates_of_change.append(l1_roc)
        l2_rates_of_change.append(l2_roc)
        h_rates_of_change.append(h_roc)
        tv_rates_of_change.append(tv_roc)

    for i in range(1, len(js_rates_of_change)):
        # plt.plot(values_of_p, js_rates_of_change[i - 1])
        # plt.plot(values_of_p[1:], no_rates_of_change[i - 1][1:])
        plt.plot(values_of_p, l1_rates_of_change[i])
        plt.plot(values_of_p, tv_rates_of_change[i])
        plt.plot(values_of_p, l2_rates_of_change[i])
        # plt.plot(values_of_p, h_rates_of_change[i - 1])
        plt.suptitle("TV-Distance(binom(p, %d), binom(p + e, %d) / e" % (num_tosses, num_tosses))
        plt.title("Blue: e = %s, Orange: e = %s" % (np.float128(epsilons[i - 1]), np.float128(epsilons[i])))
        plt.xlabel("p")
        plt.ylabel("see title")
        plt.show()

# intervals [[0, 1], [0, 1]]
def simple_parametrization_for_3_dims(params):
    height = params[0]
    point_one = np.array([1.0 - height, bigfloat.BigFloat(0.0), height])
    point_two = np.array([bigfloat.BigFloat(0.0), 1.0 - height, height])
    return params[1] * point_two + (1.0 - params[1]) * point_one

# intervals[[0, bigfloat.const_pi() / 2], [0, 1]]
def radial_parametrization_for_3_dims(params):
    angle = params[0]
    assert angle <= bigfloat.const_pi() / 2.0
    percent_radius = params[1]
    assert percent_radius <= 1.0

    p1 = bigfloat.cos(angle)
    p2 = bigfloat.sin(angle)
    p1_scaled = p1 / (p1 + p2)
    p2_scaled = p2 / (p1 + p2)
    assert float(p1_scaled + p2_scaled) == 1.0

    bottom = np.array([p1_scaled, p2_scaled, bigfloat.BigFloat(0.0)])
    top = np.array([bigfloat.BigFloat(0.0), bigfloat.BigFloat(0.0), bigfloat.BigFloat(1.0)])
    return percent_radius * bottom + (1.0 - percent_radius) * top

def compare_various_uniforms(metric="TV"):
    print("First Method...")
    if metric == "TV":
        method_1 = [random_L1_dist_over_n_elements(3) for i in range(0, 10000)]
        dm = total_variation_distance
    elif metric == "L2":
        method_1 = [random_L2_dist_over_n_elements(3) for i in range(0, 10000)]
        dm = L2_distance
    else:
        assert metric == "H"
        method_1 = [random_Hellinger_dist_over_n_elements(3) for i in range(0, 10000)]
        dm = hellinger_distance
    # print([[float(v) for v in x] for x in method_1])

    print("Second Method...")
    (points, dist) = uniform_dist_over_parametrized_credal_set(\
        param_intervals=[[bigfloat.BigFloat(0.0), bigfloat.BigFloat(1.0)], \
                         [bigfloat.BigFloat(0.0), bigfloat.BigFloat(1.0)]], \
        params_to_dist=simple_parametrization_for_3_dims, \
        distance_metric=dm, \
        num_points_per_param=401)
    method_2 = samples_from_discrete_dist(points, dist, 10000)
    # print([float(v) for v in dist])
    # print([[float(v) for v in x] for x in method_2])

    print("Third Method...")
    (points, dist) = uniform_dist_over_parametrized_credal_set(\
        param_intervals=[[bigfloat.BigFloat(0.0), bigfloat.const_pi() / 2.0], \
                         [bigfloat.BigFloat(0.0), bigfloat.BigFloat(1.0)]], \
        params_to_dist=radial_parametrization_for_3_dims, \
        distance_metric=dm, \
        num_points_per_param=401)
    method_3 = samples_from_discrete_dist(points, dist, 10000)
    # print([float(v) for v in dist])
    # print([[float(v) for v in x] for x in method_3])

    fig = plt.figure()
    # The 111 is necessary for some weird reason
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.scatter([np.float64(x[0]) for x in method_1], [np.float64(x[1]) for x in method_1], [np.float64(x[2]) for x in method_1], marker='o', alpha=0.02)
    if metric == "H":
        plt.title("Calafiore's L2-sphere --> Hellinger")
    elif metric == "TV":
        plt.title("L1 (TV) Uniform")
    elif metric == "L2":
        plt.title("L2 Uniform")
    # plt.show()
    plt.savefig('figures/%s_Uniform_Method_1.png' % metric)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.scatter([np.float64(x[0]) for x in method_2], [np.float64(x[1]) for x in method_2], [np.float64(x[2]) for x in method_2], marker='o', alpha=0.02)
    plt.title("Simple Parametrization")
    # plt.show()
    plt.savefig('figures/%s_Uniform_Method_2.png' % metric)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.scatter([np.float64(x[0]) for x in method_3], [np.float64(x[1]) for x in method_3], [np.float64(x[2]) for x in method_3], marker='o', alpha=0.02)
    plt.title("Simple Parametrization")
    # plt.show()
    plt.savefig('figures/%s_Uniform_Method_3.png' % metric)
    plt.close()

def compare_uniform_generation():
    L1_points = [random_L1_dist_over_n_elements(3) for i in range(0, 5000)]
    L2_points = [random_L2_dist_over_n_elements(3) for i in range(0, 5000)]
    H_points = [random_Hellinger_dist_over_n_elements(3) for i in range(0, 5000)]

    fig = plt.figure()
    # The 111 is necessary for some weird reason
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.scatter([np.float64(x[0]) for x in L1_points], [np.float64(x[1]) for x in L1_points], [np.float64(x[2]) for x in L1_points], marker='o', alpha=0.02)
    plt.title("L1 Uniform")
    plt.show()
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.scatter([np.float64(x[0]) for x in L2_points], [np.float64(x[1]) for x in L2_points], [np.float64(x[2]) for x in L2_points], marker='o', alpha=0.02)
    plt.title("L2 Uniform")
    plt.show()
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.scatter([np.float64(x[0]) for x in H_points], [np.float64(x[1]) for x in H_points], [np.float64(x[2]) for x in H_points], marker='o', alpha=0.02)
    plt.title("Hellinger Uniform???? Maybe???")
    plt.show()

# Conclusion: TV(Ci(x), Ci(y)) =/= integral_{x to y} (lim e -> 0 TV(Ci(p), Ci(p + e)) / e) dp
def compare_so_called_derivatives_to_integral():

    binomial_10_tosses = (lambda n : (lambda p : binomial_dist(n, p[0])))(10)

    num_binomials = 201
    print("Generating %d Binomials and 'Derivatives'..." % num_binomials)

    param_min_and_max = [bigfloat.exp2(-20), \
                         1.0 - bigfloat.exp2(-20)]

    (distributions, param_derivs) = \
        singly_parametrized_dists_and_derivatives(\
            alt_dist_generator=binomial_10_tosses, \
            alt_dist_param_bounds = param_min_and_max, \
            num_param_options=num_binomials)

    print("  Finished Generating Binomials and 'Derivatives.'")

    print("Generating %d Pairwise Distances..." % (num_binomials * (num_binomials - 1)))
    pairwise_distances = {}
    for i in range(0, num_binomials):
        for j in range(i + 1, num_binomials):
            pairwise_distances[(i, j)] = total_variation_distance(\
                distributions[i], distributions[j])
    print("  Finished Generating Pairwise Distances.")

    print("Generating Cumulative Sums...")
    c = bigfloat.BigFloat(0)
    param_cumulatives = []
    for i in range(0, num_binomials):
        c += param_derivs[i]
        param_cumulatives.append(c)
    print("  Finished Generating Cumulative Sums.")

    print("Generating %d Pairwise 'Integrals'..." % (num_binomials * (num_binomials - 1)))
    pairwise_integrals = {}
    diff_ratios = []
    dp_di = (param_min_and_max[1] - param_min_and_max[0]) / (num_binomials - 1)
    for i in range(0, num_binomials):
        for j in range(i + 1, num_binomials):
            pairwise_integrals[(i, j)] = (param_cumulatives[j] - param_cumulatives[i]) * dp_di
            diff_ratio = bigfloat.abs(pairwise_integrals[(i, j)] - pairwise_distances[(i, j)]) / \
                  (pairwise_integrals[(i, j)] + pairwise_distances[(i, j)])
            if j == (num_binomials - 1) and i == 0:
                print("    %s" % pairwise_distances[(i, j)])
                print("    vs")
                print("    %s" % pairwise_integrals[(i, j)])
            diff_ratios.append((diff_ratio, (float(i) / (num_binomials - 1), float(j) / (num_binomials - 1))))
    diff_ratios.sort(reverse=True)
    print("  Finished Generating Pairwise 'Integrals.'")

    print(diff_ratios[0])
    print(diff_ratios[1])
    print(diff_ratios[2])

def __bin_diameter_finder_helper__(p, bg, bin_a, bin_b):
    bin_p = bg(p)
    # return 2.0 * bigfloat.max(hellinger_distance(bin_a, bin_p), \
    #                           hellinger_distance(bin_b, bin_p))
    return 2.0 * bigfloat.max(total_variation_distance(bin_a, bin_p), \
                              total_variation_distance(bin_b, bin_p))

def find_diameter_of_binomials_ball(binomial_generator, a, b):
    bin_a = binomial_generator(a)
    bin_b = binomial_generator(b)

    func = (lambda args: (lambda p: __bin_diameter_finder_helper__(p, args[0], args[1], args[2])))((binomial_generator, bin_a, bin_b))
    
    (best_arg, diameter) = optimizing.binary_min_finder(func, a, b, tol=bigfloat.exp2(-40), error_depth=1)
    return diameter

def __bin_split_point_finder_helper__(p, bg, a, b):
    return bigfloat.max(find_diameter_of_binomials_ball(bg, a, p), \
                        find_diameter_of_binomials_ball(bg, p, b))

def find_split_point_of_binomials_ball(binomial_generator, a, b):

    func = (lambda args: (lambda p: __bin_split_point_finder_helper__(p, args[0], args[1], args[2])))((binomial_generator, a, b))

    (split_point, best_func) = optimizing.binary_min_finder(func, a, b, tol=bigfloat.exp2(-40), error_depth=1)
    return split_point

def test_uniformity_idea_existence_on_binomials():
    binomial_generator = (lambda n : (lambda p : binomial_dist(n, p)))(1)
    zero_mark = bigfloat.BigFloat(0.0)
    half_mark = bigfloat.BigFloat(0.5)
    full_mark = bigfloat.BigFloat(1.0)

    first_half_diam = find_diameter_of_binomials_ball(binomial_generator, zero_mark, half_mark)
    second_half_diam = find_diameter_of_binomials_ball(binomial_generator, half_mark, full_mark)

    # Num characters
    chars = 12

    print("Sanity Check: %s == %s ? (should be yes)" % \
        (str(first_half_diam)[:chars], str(second_half_diam)[:chars]))

    quarter_mark = \
        find_split_point_of_binomials_ball(binomial_generator, zero_mark, half_mark)

    three_fourths_mark = \
        find_split_point_of_binomials_ball(binomial_generator, half_mark, full_mark)

    test_diam_one = \
        find_diameter_of_binomials_ball(binomial_generator, zero_mark, quarter_mark)

    test_diam_two = \
        find_diameter_of_binomials_ball(binomial_generator, quarter_mark, half_mark)

    test_diam_three = \
        find_diameter_of_binomials_ball(binomial_generator, half_mark, three_fourths_mark)

    test_diam_four = \
        find_diameter_of_binomials_ball(binomial_generator, three_fourths_mark, full_mark)

    print("All the Quarters:")
    print("  %s == %s == %s == %s ?" % \
        (str(test_diam_one)[:chars], str(test_diam_two)[:chars], \
         str(test_diam_three)[:chars], str(test_diam_four)[:chars]))

    eighth_mark = \
        find_split_point_of_binomials_ball(binomial_generator, zero_mark, quarter_mark)

    three_eighths_mark = \
        find_split_point_of_binomials_ball(binomial_generator, quarter_mark, half_mark)

    another_quarter_diam = find_diameter_of_binomials_ball(binomial_generator, eighth_mark, three_eighths_mark)
    print("Another 'Quarter': %s" % (str(another_quarter_diam)[:chars]))

    test_diam_one = \
        find_diameter_of_binomials_ball(binomial_generator, zero_mark, eighth_mark)

    test_diam_two = \
        find_diameter_of_binomials_ball(binomial_generator, eighth_mark, quarter_mark)

    test_diam_three = \
        find_diameter_of_binomials_ball(binomial_generator, quarter_mark, three_eighths_mark)

    test_diam_four = \
        find_diameter_of_binomials_ball(binomial_generator, three_eighths_mark, half_mark)

    print("First Half of the Eighths:")
    print("  %s == %s == %s == %s ?" % \
        (str(test_diam_one)[:chars], str(test_diam_two)[:chars], \
         str(test_diam_three)[:chars], str(test_diam_four)[:chars]))

    sixteenth_mark = \
        find_split_point_of_binomials_ball(binomial_generator, zero_mark, eighth_mark)

    three_sixteenths_mark = \
        find_split_point_of_binomials_ball(binomial_generator, eighth_mark, quarter_mark)

    five_sixteenths_mark = \
        find_split_point_of_binomials_ball(binomial_generator, quarter_mark, three_eighths_mark)

    seven_sixteenths_mark = \
        find_split_point_of_binomials_ball(binomial_generator, three_eighths_mark, half_mark)

    test_diam_one = \
        find_diameter_of_binomials_ball(binomial_generator, zero_mark, sixteenth_mark)

    test_diam_two = \
        find_diameter_of_binomials_ball(binomial_generator, sixteenth_mark, eighth_mark)

    test_diam_three = \
        find_diameter_of_binomials_ball(binomial_generator, eighth_mark, three_sixteenths_mark)

    test_diam_four = \
        find_diameter_of_binomials_ball(binomial_generator, three_sixteenths_mark, quarter_mark)

    test_diam_five = \
        find_diameter_of_binomials_ball(binomial_generator, quarter_mark, five_sixteenths_mark)

    test_diam_six = \
        find_diameter_of_binomials_ball(binomial_generator, five_sixteenths_mark, three_eighths_mark)

    test_diam_seven = \
        find_diameter_of_binomials_ball(binomial_generator, three_eighths_mark, seven_sixteenths_mark)

    test_diam_eight = \
        find_diameter_of_binomials_ball(binomial_generator, seven_sixteenths_mark, half_mark)

    print("First Half of the Sixteenths:")
    print("  %s == %s == %s == %s ?" % \
        (str(test_diam_one)[:chars], str(test_diam_two)[:chars], \
         str(test_diam_three)[:chars], str(test_diam_four)[:chars]))
    print("  %s == %s == %s == %s ?" % \
        (str(test_diam_five)[:chars], str(test_diam_six)[:chars], \
         str(test_diam_seven)[:chars], str(test_diam_eight)[:chars]))

    print("")
    print("For curiosity's sake, here are the splits:")
    print("%f - %f - %f - %f - %f - %f - %f - %f - %f" % (zero_mark, \
        sixteenth_mark, eighth_mark, three_sixteenths_mark, quarter_mark, \
        five_sixteenths_mark, three_eighths_mark, seven_sixteenths_mark, \
        half_mark))

    print("So... jury is still out?")

if __name__ == "__main__":
    # uniform_calcs_for_urn_sampling(90, 3)

    bf_context = bigfloat.Context(precision=2000, emax=100000000, emin=-100000000)
    bigfloat.setcontext(bf_context)

    # binomial_likelihood_function_plot(n=100, c_values_to_plot=[0, 10, 40, 50], num_binoms=1001, metric="TV")
    # exit(0)
    # compare_various_uniforms(metric="TV")
    # exit(0)

    test_for_higher_order_convergence_with_binomials(null_p=0.5, \
        coin_tosses=10, heads=[1, 3, 5], \
        num_dists_by_order=[16000, 16000, 16000, 16000, 16000], \
        order_names=["First", "Second", "Third", "Fourth", "Fifth"], \
        metric="TV")
    exit(0)

    test_uniformity_idea_existence_on_binomials()
    exit(0)

    test_for_higher_order_convergence_with_binomials()
    exit(0)

    # compare_L1_and_L2_dist_generation()
    # exit(0)

    test_distance_metrics_for_linearity_of_immediate_space_on_binomials()
    exit(0)
    test_for_a_natural_distance_metric()
