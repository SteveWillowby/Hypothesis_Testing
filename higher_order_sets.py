from comb_and_prob_funcs import *
import random
import numpy as np
import bigfloat
import matplotlib.pyplot as plt

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
    log2_A = np.array([bigfloat.log2(v) for v in div_A])
    log2_B = np.array([bigfloat.log2(v) for v in div_B])
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

def total_variation_distance_for_binomials(binom_A, binom_B):
    zeros = np.array([bigfloat.BigFloat(0.0) for v in binom_A])
    v1 = np.sum(np.maximum(np.subtract(binom_A, binom_B), zeros))
    v2 = np.sum(np.maximum(np.subtract(binom_B, binom_A), zeros))
    return bigfloat.max(v1, v2)

def random_dist_over_n_elements(n):
    # Use uniform between 0 and n rather than 0 and 1 to HOPEFULLY allow more
    # precision.
    rng = np.random.default_rng()
    basic_numbers = [np.float64(0.0)] + \
        [rng.uniform(0.0, n) for i in range(0, n - 1)] + [np.float64(n)]
    basic_numbers.sort()
    differences = np.array([basic_numbers[i + 1] - basic_numbers[i] for i in range(0, n)], bigfloat.BigFloat)
    return differences / n

# Rather than returning the full new dist, returns the implied dist over the
# lower-level sample space.
def generate_random_dist_over_dists(basic_dists_transposed):
    dist_over_dists = random_dist_over_n_elements(len(basic_dists_transposed[0]))

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

def binomial_dist(n, p):
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

def test_for_higher_order_convergence_on_single_binomial(null_p=0.5, \
        coin_tosses=1000, heads=400, \
        num_binoms=101, num_priors=101, \
        num_higher_order_dists=[10000, 10000], \
        higher_order_names=["Second", "Third"]):

    print("Creating Null Dist")
    null_dist = \
        np.array([bigfloat_prob_of_count_given_p(c, null_p, coin_tosses) for \
            c in range(0, coin_tosses + 1)])
    print("  Null Dist Complete")

    print("Creating Alternate Dists")
    alternate_dists = \
        np.array(m_binomial_dists_over_n_coin_tosses(m=num_binoms, n=coin_tosses))
    print("  Alternate Dists Complete")

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
    print("Combining Dists Complete")

    # Now P(flip c heads) = dist[c] + dist[coin_tosses + c]
    # P(flip c heads | N) = dist[c]
    # P(flip c heads | not-N) = dist[coin_tosses + c]
    # P(N) = sum( dist[0...coin_tosses) )
    # P(not-N) = sum( dist[coin_tosses...end) )

    print("Getting Likelihood Ratios")
    likelihood_ratios = [dist[coin_tosses + heads] / dist[heads] for \
                            dist in full_dist_collection]
    print("  Getting Likelihod Ratios Complete")

    print("Plotting Likelihood Ratios")
    plot_log_of_likelihood_ratios(likelihood_ratios, coin_tosses, \
        heads, order="First")
    print("  Plotting Likelihood Ratios Complete")

    new_dists = full_dist_collection
    for order_idx in range(0, len(higher_order_names)):
        old_dists_transposed = new_dists.transpose()
        num_dists = num_higher_order_dists[order_idx]
        order_name = higher_order_names[order_idx]
        new_dists = []
        print("Working on %s Order Prob Functions" % order_name)
        for i in range(0, num_dists):
            if (i % (num_dists / 100)) == 0:
                print("    %d percent done" % (i / (num_dists / 100)))
            new_dists.append(generate_random_dist_over_dists(old_dists_transposed))
        new_dists = np.array(new_dists)
        print("  Accumulated %s Order Prob Functions" % order_name)
        
        print("Getting Likelihood Ratios and Plotting")
        likelihood_ratios = [dist[coin_tosses + heads] / dist[heads] for \
                                dist in new_dists]

        plot_log_of_likelihood_ratios(likelihood_ratios, coin_tosses, \
            heads, order=order_name)
        print("  Obtained Likelihood Ratios and Plotted")

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
    epsilon_exponents = [-16, -32, -64, -128]
    epsilons = [bigfloat.exp2(v) for v in epsilon_exponents]

    num_tosses = 100
    start_p = bigfloat.BigFloat(0.0)
    end_p = bigfloat.BigFloat(0.5)
    num_p = 101

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
    h_rates_of_change = []
    tv_rates_of_change = []
    for e in epsilons:
        js_roc = []
        no_roc = []
        h_roc = []
        tv_roc = []
        for base_p in values_of_p:
            next_p = base_p + e
            js_roc.append(jensen_shannon_distance(binomials[base_p], binomials[next_p]) / e)
            no_roc.append(naive_overlap_distance(binomials[base_p], binomials[next_p]) / e)
            h_roc.append(hellinger_distance(binomials[base_p], binomials[next_p]) / e)
            tv_roc.append(total_variation_distance_for_binomials(binomials[base_p], binomials[next_p]) / e)
        js_rates_of_change.append(js_roc)
        no_rates_of_change.append(no_roc)
        h_rates_of_change.append(h_roc)
        tv_rates_of_change.append(tv_roc)

    for i in range(1, len(js_rates_of_change)):
        # plt.plot(values_of_p, js_rates_of_change[i - 1])
        # plt.plot(values_of_p, no_rates_of_change[i - 1])
        plt.plot(values_of_p, h_rates_of_change[i - 1])
        # plt.plot(values_of_p, tv_rates_of_change[i - 1])
        plt.suptitle("JS-Distance(binom(p, %d), binom(p + e, %d) / e" % (num_tosses, num_tosses))
        plt.title("Blue: e = %s, Orange: e = %s" % (np.float128(epsilons[i - 1]), np.float128(epsilons[i])))
        plt.xlabel("p")
        plt.ylabel("see title")
        plt.show()

if __name__ == "__main__":
    bf_context = bigfloat.Context(precision=2000, emax=100000, emin=-100000)
    bigfloat.setcontext(bf_context)
    test_distance_metrics_for_linearity_of_immediate_space_on_binomials()
    exit(0)
    test_for_a_natural_distance_metric()

    test_for_higher_order_convergence_on_single_binomial(null_p=0.5, \
        coin_tosses=100, heads=40, \
        num_binoms=31, num_priors=31, \
        num_higher_order_dists=[1000, 1000, 1000], \
        higher_order_names=["Second", "Third", "Fourth"])
