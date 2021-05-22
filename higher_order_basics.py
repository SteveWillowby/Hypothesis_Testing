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
        if next_prob > 1.0:
            print("Prob too large! %f" % next_prob)
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


# Returns the distributions in a LIST (list of np-arrays) as well as the
#   distribution over them in an np-array.
#
# params_to_dist should take a single LIST of param values.
def uniform_dist_over_parametrized_credal_set(\
        param_intervals, params_to_dist, \
        distance_metric=total_variation_distance, \
        num_points_per_param=1001):
    
    num_params = len(param_intervals)
    idxs = [0 for _ in range(0, num_params)]
    param_mins = [param_intervals[i][0] for i in range(0, num_params)]
    param_maxs = [param_intervals[i][1] for i in range(0, num_params)]
    param_midpoints = [(param_maxs[i] - param_mins[i]) / 2.0 for \
                            i in range(0, num_params)]
    param_increments = [(param_maxs[i] - param_mins[i]) / \
                            (num_points_per_param - 1) for \
                                i in range(0, num_params)]

    epsilons = [bigfloat.exp2(-128) * (param_maxs[i] - param_mins[i]) / \
                    num_points_per_param for i in range(0, num_params)]

    distributions = []
    measure = []

    done = False
    while not done:
        print(idxs)
        point = [param_mins[i] + idxs[i] * param_increments[i] for \
                    i in range(0, num_params)]

        dist = params_to_dist(point)
        distributions.append(dist)

        full_product = bigfloat.BigFloat(1.0)
        for i in range(0, num_params):
            sign = 1.0
            if point[i] > param_midpoints[i]:
                sign = -1.0
            alt_point = list(point)
            alt_point[i] += sign * epsilons[i]
            alt_dist = params_to_dist(alt_point)

            full_product *= distance_metric(dist, alt_dist) / epsilons[i]

        measure.append(full_product)

        increment_idx = num_params - 1
        idxs[increment_idx] += 1
        while idxs[increment_idx] == num_points_per_param:
            idxs[increment_idx] = 0
            increment_idx -= 1
            if increment_idx < 0:
                done = True
                break
            idxs[increment_idx] += 1

    measure = np.array(measure)
    measure = measure / np.sum(measure)

    return (distributions, measure)

def pdf_to_cdf(pdf):
    cdf = []
    cp = bigfloat.BigFloat(0.0)
    for p in pdf:
        cp += p
        cdf.append(cp)
    cdf[-1] = bigfloat.BigFloat(1.0)
    return np.array(cdf)

# Runs in O(num_samples * log(num_samples))
def samples_from_discrete_dist(elements, dist, num_samples):
    global __higher_order_reference_rng__
    if __higher_order_reference_rng__ is None:
        __higher_order_reference_rng__ = np.random.default_rng()
    rng = __higher_order_reference_rng__

    sample_weights = rng.uniform(0.0, num_samples, num_samples)
    sample_weights = [bigfloat.BigFloat(x) / num_samples for x in sample_weights]
    sample_weights.sort()

    cdf = pdf_to_cdf(dist)

    samples = []
    sw_idx = 0
    element_idx = 0
    while sw_idx < num_samples:
        if sample_weights[sw_idx] < cdf[element_idx]:
            samples.append(elements[element_idx])
            sw_idx += 1
        else:
            element_idx += 1

    return samples
