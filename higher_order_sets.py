from comb_and_prob_funcs import *
import random
import numpy as np
import optimizing
import bigfloat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#TODO: Check if these are well-defined on infinite sets and prob functions over
#       those sets, where the prob function uses this distance as the metric for
#       the Borel space.

def jensen_shannon_distance(dist_A, dist_B):
    mid = 0.5 * dist_A + 0.5 * dist_B
    # Get KL divergences with mid distribution - Use log base 2 to bound things
    #   to be one at most

    # The following log2 does not seem to work with bigfloats
    # KL_A = np.sum(np.multiply(dist_A, np.log2(np.divide(dist_A, mid))))
    # KL_B = np.sum(np.multiply(dist_B, np.log2(np.divide(dist_B, mid))))

    div_A = np.divide(dist_A, mid)
    div_B = np.divide(dist_B, mid)
    log2_A = np.array([bigfloat.log2(v) if v > bigfloat.BigFloat(0.0) else bigfloat.BigFloat(1.0) for v in div_A])
    log2_B = np.array([bigfloat.log2(v) if v > bigfloat.BigFloat(0.0) else bigfloat.BigFloat(1.0) for v in div_B])
    KL_A = np.sum(np.multiply(dist_A, log2_A))
    KL_B = np.sum(np.multiply(dist_B, log2_B))

    # Take the square root to convert from JS divergence to JS distance
    #   (the latter is a metric but the former is not)
    return bigfloat.sqrt(0.5 * KL_A + 0.5 * KL_B)

def naive_overlap_distance(dist_A, dist_B):
    return 1.0 - np.sum(np.minimum(dist_A, dist_B))

def hellinger_distance(dist_A, dist_B):
    BC = np.sum([bigfloat.sqrt(v) for v in np.multiply(dist_A, dist_B)])
    return bigfloat.sqrt(1.0 - BC)

def L1_distance(dist_A, dist_B):
    return np.sum(np.abs(dist_A - dist_B))

def L2_distance(dist_A, dist_B):
    return bigfloat.sqrt(np.sum(np.square(dist_A - dist_B)))

def total_variation_distance(dist_A, dist_B):
    zeros = np.array([bigfloat.BigFloat(0.0) for v in dist_A])
    v1 = np.sum(np.maximum(np.subtract(dist_A, dist_B), zeros))
    v2 = np.sum(np.maximum(np.subtract(dist_B, dist_A), zeros))
    return bigfloat.max(v1, v2)

# Only instantiate an RNG once.
__higher_order_reference_rng__ = None

def random_L2_dist_over_n_elements(n):
    # Use uniform between 0 and n rather than 0 and 1 to HOPEFULLY allow more
    # precision.
    global __higher_order_reference_rng__
    if __higher_order_reference_rng__ is None:
        __higher_order_reference_rng__ = np.random.default_rng()
    rng = __higher_order_reference_rng__

    basic_numbers = [np.float64(0.0)] + \
        [rng.uniform(0.0, n) for i in range(0, n - 1)] + [np.float64(n)]
    basic_numbers.sort()
    differences = np.array([basic_numbers[i + 1] - basic_numbers[i] for i in range(0, n)], bigfloat.BigFloat)
    return differences / n

def random_L1_dist_over_n_elements(n):
    global __higher_order_reference_rng__
    if __higher_order_reference_rng__ is None:
        __higher_order_reference_rng__ = np.random.default_rng()
    rng = __higher_order_reference_rng__
    raw_coords = rng.gamma(1.0, 1.0, n)
    raw_coords = np.array([bigfloat.BigFloat(x) for x in raw_coords])
    return raw_coords / np.sum(raw_coords)

# Rather than returning the full new dist, returns the implied dist over the
# lower-level sample space.
def generate_random_dist_over_dists(basic_dists_transposed):
    dist_over_dists = random_L1_dist_over_n_elements(len(basic_dists_transposed[0]))

    scale_rows_by_meta_dist = basic_dists_transposed * dist_over_dists
    collapsed = np.sum(scale_rows_by_meta_dist, axis=1)

    """
    dist_over_original_sample_space = []
    for space_idx in range(0, len(basic_dists[0])):
        total = bigfloat.BigFloat(0.0)
        for dist_idx in range(0, len(dist_over_dists)):
            total += basic_dists[dist_idx][space_idx] * \
                     dist_over_dists[dist_idx]
        dist_over_original_sample_space.append(total)
    dist_over_original_sample_space = np.array(dist_over_original_sample_space)
    """
    return collapsed

def collapse_dist_to_implied(basic_dists_transposed, dist_over_dists):
    scale_rows_by_meta_dist = basic_dists_transposed * dist_over_dists
    collapsed = np.sum(scale_rows_by_meta_dist, axis=1)
    return collapsed

__cached_binomial_dists__ = {}

def caching_binomial_dist(n, p):
    global __cached_binomial_dists__
    if (n, p) not in __cached_binomial_dists__:
        __cached_binomial_dists__[(n, p)] = binomial_dist(n, p)
    return __cached_binomial_dists__[(n, p)]

def binomial_dist(n, p):
    if p == bigfloat.BigFloat(0.0):
        return np.array([bigfloat.BigFloat(1.0)] + \
            [bigfloat.BigFloat(0.0) for i in range(0, n)])
    elif p == bigfloat.BigFloat(1.0):
        return np.array([bigfloat.BigFloat(0.0) for i in range(0, n)] + \
            [bigfloat.BigFloat(1.0)])
    p_ratio = p / (1.0 - p)
    dist = []
    next_prob = bigfloat.pow(1.0 - p, n)  # prob of zero heads
    for h in range(0, n + 1):
        dist.append(next_prob)
        next_prob = (next_prob * p_ratio * (n - h)) / (h + 1)
    return np.array(dist)

def m_binomial_dists_over_n_coin_tosses(m, n, start_p=bigfloat.exp2(-10.0)):
    p = start_p
    p_inc = bigfloat.BigFloat(1.0 - (2.0 * start_p)) / (m - 1)
    dists = []
    for _ in range(0, m):
        dists.append(binomial_dist(n, p))
        p += p_inc
    return dists

def plot_log_of_likelihood_ratios(likelihood_ratios, coin_tosses, \
        heads, order="First"):
    ratios = list(likelihood_ratios)
    ratios.sort()
    dist_idx_ratio = [bigfloat.BigFloat(i) / (len(ratios) - 1) for \
                        i in range(0, len(ratios))]

    ratios = [bigfloat.log2(r) for r in ratios]
    plt.plot(dist_idx_ratio, ratios)
    plt.suptitle("Evidence Against/For Null for %s Order Probs" % order)
    plt.title("For %d heads on %d coin tosses" % (heads, coin_tosses))
    plt.xlabel("Different Probability Functions")
    plt.ylabel("Log_2 of Likelihood Ratio Alternative/Null")
    plt.savefig("coins_%d_of_%d_%s_order.pdf" % (heads, coin_tosses, order))
    plt.close()

def singly_parametrized_dists_and_derivatives(\
        alt_dist_generator, alt_dist_param_bounds, \
        num_param_options=1001):

    param_min = alt_dist_param_bounds[0]
    param_max = alt_dist_param_bounds[1]

    param_inc = (param_max - param_min) / (num_param_options - 1)

    epsilon = bigfloat.exp2(-128)

    distributions = []
    param_derivative_multiples = []
    for i in range(0, num_param_options):
        param = param_min + ((param_max - param_min) * i) / \
                                (num_param_options - 1)
        alt_dist = alt_dist_generator(param)
        distributions.append(alt_dist)

        if param > param_min + (param_max - param_min) / 2:
            deriv_param = param - epsilon
        else:
            deriv_param = param + epsilon

        deriv_dist = alt_dist_generator(deriv_param)
        param_derivative_multiples.append(\
            total_variation_distance(alt_dist, deriv_dist) / epsilon)

    param_derivative_multiples = np.array(param_derivative_multiples)

    return (distributions, param_derivative_multiples)

def representative_sampling_of_singly_parametrized_dists(\
        num_samples, alt_dist_generator, alt_dist_param_bounds, \
        num_param_options=1001):

    (distributions, param_derivative_multiples) = \
        singly_parametrized_dists_and_derivatives(alt_dist_generator, \
            alt_dist_param_bounds, num_param_options)

    param_probabilities = param_derivative_multiples / \
                            np.sum(param_derivative_multiples)

    global __higher_order_reference_rng__
    if __higher_order_reference_rng__ is None:
        __higher_order_reference_rng__ = np.random.default_rng()
    rng = __higher_order_reference_rng__

    dists_weights = rng.uniform(0.0, num_samples, num_samples)
    dists_weights = [bigfloat.BigFloat(x) / num_samples for x in dists_weights]
    dists_weights.sort()

    sampled_dists = []

    cumulative_param_probs = []
    cp = bigfloat.BigFloat(0.0)
    for p in param_probabilities:
        cp += p
        cumulative_param_probs.append(cp)
    cumulative_param_probs[-1] = bigfloat.BigFloat(1.0)

    dw_idx = 0
    param_idx = 0
    while dw_idx < num_samples:
        if dists_weights[dw_idx] < cumulative_param_probs[param_idx]:
            sampled_dists.append(distributions[param_idx])
            dw_idx += 1
        else:
            param_idx += 1

    return sampled_dists

def test_for_higher_order_convergence_with_binomials(null_p=0.5, \
        coin_tosses=50, heads=20, \
        num_dists_by_order=[10000, 5000, 2500, 1250], \
        order_names=["First", "Second", "Third", "Fourth"]):

    binomial = (lambda n : (lambda p : binomial_dist(n, p)))(coin_tosses)

    print("Creating Null Dist")
    null_dist = binomial_dist(coin_tosses, null_p)
    print("  Null Dist Complete")

    # If ensure_null_present is true, then the sample is biased and not QUITE
    #   representative.
    ensure_null_present = False
    done = False
    while not done:
        print("Creating Representative First Order Dist Sample")
        first_order_dists = \
            representative_sampling_of_singly_parametrized_dists(
                num_samples=num_dists_by_order[0], \
                alt_dist_generator=binomial, \
                alt_dist_param_bounds = [bigfloat.exp2(-20), \
                                         1.0 - bigfloat.exp2(-20)], \
                num_param_options=(num_dists_by_order[0] * 10 + 1))

        print("  Sample Complete")
        if ensure_null_present:
            for dist in first_order_dists:
                if dist.all() == null_dist.all():
                    done = True
                    break
            if not done:
                print("Issue! Didn't get null dist in 1st order dists - retrying.")
        else:
            done = True

    print("Generating Uniform Second Order Dist")
    uniform_second_order_dist = first_order_dists[0]
    for i in range(1, len(first_order_dists)):
        uniform_second_order_dist += first_order_dists[i]
    uniform_second_order_dist /= len(first_order_dists)
    print("  Generating Uniform Second Order Dist Complete")

    print("Plotting Uniform Second Order Dist")
    plt.plot([i for i in range(0, coin_tosses + 1)], uniform_second_order_dist)
    plt.title("Dist Over Num Heads Implied by Uniform Second Order Dist")
    plt.savefig("second_order_uniform_over_heads.pdf")
    plt.close()
    print("  Plotting of Uniform Second Order Dist Complete")

    """
    print("Combining Dists")
    full_dist_collection = []
    prior = bigfloat.exp2(-10.0)
    prior_inc = (1.0 - (2.0 * prior)) / (num_priors - 1)
    for prior_idx in range(0, num_priors):
        for alternate_dist in alternate_dists:
            # Make a JOINT probability space with N and not-N as events
            full_dist_collection.append(np.concatenate((prior * null_dist, \
                                            (1.0 - prior) * alternate_dist)))
        prior += prior_inc
    full_dist_collection = np.array(full_dist_collection)
    print("  Combining Dists Complete")

    # Now P(flip c heads) = dist[c] + dist[coin_tosses + c]
    # P(flip c heads | N) = dist[c]
    # P(flip c heads | not-N) = dist[coin_tosses + c]
    # P(N) = sum( dist[0...coin_tosses) )
    # P(not-N) = sum( dist[coin_tosses...end) )
    """

    print("Getting Chances of %d Heads from %d Tosses" % (heads, coin_tosses))
    first_order_chances = [dist[heads] for dist in first_order_dists]
    first_order_chances.sort()
    print("  Getting Chances Complete")

    orders_chances = [first_order_chances]

    new_dists = np.array(first_order_dists)
    for order_idx in range(1, len(num_dists_by_order)):
        old_dists_transposed = new_dists.transpose()
        num_dists = num_dists_by_order[order_idx]
        order_name = order_names[order_idx]
        new_dists = []
        print("Working on %s Order Prob Functions" % order_name)
        for i in range(0, num_dists):
            new_dists.append(generate_random_dist_over_dists(old_dists_transposed))
        new_dists = np.array(new_dists)
        print("  Accumulated %s Order Prob Functions" % order_name)

        order_chances = [dist[heads] for dist in new_dists]
        order_chances.sort()
        orders_chances.append(order_chances)

    print("Plotting Ordered Chances of %d heads from %d Tosses" % \
            (heads, coin_tosses))
    for i in range(0, len(orders_chances)):
        order_chances = orders_chances[i]
        x_axis = [bigfloat.BigFloat(j) / (len(order_chances) - 1) \
            for j in range(0, len(order_chances))]
        plt.plot(x_axis, order_chances)

    plt.plot([0, 1], [uniform_second_order_dist[heads], uniform_second_order_dist[heads]], linestyle="dashed")

    suptitle = "Representative Chances of %d Heads on %d Tosses" % \
        (heads, coin_tosses)
    title = "For"
    for i in range(0, len(order_names) - 1):
        title += " %s," % order_names[i]
    title += ", and %s Order Confidences" % order_names[-1]
    plt.suptitle(suptitle)
    plt.title(title)
    plt.xlabel("Just Indexing Prob Functions...")
    plt.ylabel("Chance of %d Heads on %d Tosses" % (heads, coin_tosses))
    plt.savefig("higher_order_convergence.pdf")
    plt.close()

    print("  Plotting Ordered Chances Complete")

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

    """
    plt.plot(priors, js_distances[0])
    plt.plot(priors, js_distances[1])
    plt.plot(priors, js_distances[2])
    plt.title("Jensen Shannon Distances of Distributions")
    plt.xlabel("Proportion Parameter")
    plt.ylabel("Distance from a Reference Distribution")
    plt.show()

    plt.plot(priors, no_distances[0])
    plt.plot(priors, no_distances[1])
    plt.plot(priors, no_distances[2])
    plt.title("Naive Overlap Distances of Distributions")
    plt.xlabel("Proportion Parameter")
    plt.ylabel("Distance from a Reference Distribution")
    plt.show()

    plt.plot(priors, h_distances[0])
    plt.plot(priors, h_distances[1])
    plt.plot(priors, h_distances[2])
    plt.title("Hellinger Distances of Distributions")
    plt.xlabel("Proportion Parameter")
    plt.ylabel("Distance from a Reference Distribution")
    plt.show()
    """

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

def compare_L1_and_L2_dist_generation():
    L1_points = [random_L1_dist_over_n_elements(3) for i in range(0, 5000)]
    L2_points = [random_L2_dist_over_n_elements(3) for i in range(0, 5000)]

    fig = plt.figure()
    # The 111 is necessary for some weird reason
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.scatter([np.float64(x[0]) for x in L1_points], [np.float64(x[1]) for x in L1_points], [np.float64(x[2]) for x in L1_points], marker='o', alpha=0.02)
    plt.show()
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.scatter([np.float64(x[0]) for x in L2_points], [np.float64(x[1]) for x in L2_points], [np.float64(x[2]) for x in L2_points], marker='o', alpha=0.02)
    plt.show()

# Conclusion: TV(Ci(x), Ci(y)) =/= integral_{x to y} (lim e -> 0 TV(Ci(p), Ci(p + e)) / e) dp
def compare_so_called_derivatives_to_integral():

    binomial_10_tosses = (lambda n : (lambda p : binomial_dist(n, p)))(10)

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
    return 2.0 * bigfloat.max(jensen_shannon_distance(bin_a, bin_p), \
                              jensen_shannon_distance(bin_b, bin_p))
    return 2.0 * bigfloat.max(total_variation_distance(bin_a, bin_p), \
                              total_variation_distance(bin_b, bin_p))

def find_diameter_of_binomials_ball(binomial_generator, a, b):
    bin_a = binomial_generator(a)
    bin_b = binomial_generator(b)

    func = (lambda args: (lambda p: __bin_diameter_finder_helper__(p, args[0], args[1], args[2])))((binomial_generator, bin_a, bin_b))
    
    (best_arg, diameter) = optimizing.binary_min_finder(func, a, b, tol=bigfloat.exp2(-120), error_depth=1)
    return diameter

def __bin_split_point_finder_helper__(p, bg, a, b):
    return bigfloat.max(find_diameter_of_binomials_ball(bg, a, p), \
                        find_diameter_of_binomials_ball(bg, p, b))

def find_split_point_of_binomials_ball(binomial_generator, a, b):

    func = (lambda args: (lambda p: __bin_split_point_finder_helper__(p, args[0], args[1], args[2])))((binomial_generator, a, b))

    (split_point, best_func) = optimizing.binary_min_finder(func, a, b, tol=bigfloat.exp2(-120), error_depth=1)
    return split_point

def test_uniformity_idea_existence_on_binomials():
    binomial_generator = (lambda n : (lambda p : binomial_dist(n, p)))(10)
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
    bf_context = bigfloat.Context(precision=20000, emax=10000, emin=-10000)
    bigfloat.setcontext(bf_context)

    test_uniformity_idea_existence_on_binomials()
    exit(0)

    test_for_higher_order_convergence_with_binomials()
    exit(0)

    # compare_L1_and_L2_dist_generation()
    # exit(0)

    test_distance_metrics_for_linearity_of_immediate_space_on_binomials()
    exit(0)
    test_for_a_natural_distance_metric()
