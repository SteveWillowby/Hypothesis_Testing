from comb_and_prob_funcs import *
import random
import numpy as np
import bigfloat
import matplotlib.pyplot as plt

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

def n_binomial_dists_over_m_coin_tosses(n, m):
    p = bigfloat.exp2(-10.0)
    p_inc = bigfloat.BigFloat(1.0 - (2.0 * p)) / (n - 1)
    dists = []
    for _ in range(0, n):
        # p is the chance of getting a heads
        p_ratio = p / (1.0 - p)
        dist = []
        # compute the probs iteratively so as to save computation on pow and
        #   choose functions
        next_prob = bigfloat.pow(1.0 - p, m) # prob of zero heads
        for c in range(0, m + 1):
            dist.append(next_prob)
            next_prob = (next_prob * p_ratio * (m - c)) / (c + 1)
        dists.append(dist)

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
        np.array(n_binomial_dists_over_m_coin_tosses(n=num_binoms, m=coin_tosses))
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

if __name__ == "__main__":
    test_for_higher_order_convergence_on_single_binomial(null_p=0.5, \
        coin_tosses=1000, heads=400, \
        num_binoms=31, num_priors=31, \
        num_higher_order_dists=[1000, 1000, 1000], \
        higher_order_names=["Second", "Third", "Fourth"])
