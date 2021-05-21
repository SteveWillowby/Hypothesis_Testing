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
        num_param_options=1001, \
        epsilon=bigfloat.exp2(-128)):

    param_min = alt_dist_param_bounds[0]
    param_max = alt_dist_param_bounds[1]

    param_inc = (param_max - param_min) / (num_param_options - 1)

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

# Returns the distributions in a LIST (list of np-arrays) as well as the
#   distribution over them in an np-array.
def uniform_dist_over_parametrized_credal_set(\
        param_intervals, params_to_dist, \
        distance_metric=total_variation_distance, \
        num_points_per_param=1001):
    
    (distributions, uniform_measure_multipls) = \
        __uniform_dist_over_parametrized_credal_set_helper__(\
            param_intervals, params_to_dist, \
            distance_metric, \
            num_points_per_param)

    uniform_measure_multiple = np.array(uniform_measure_multiple)
    uniform_measure = uniform_measure_multiple / \
                        np.sum(uniform_measure_multiple)

    return (distributions, uniform_measure)

# Returns the distributions in LIST as well as the un-normalized uniform
#   measure over the space in a LIST.
#
# param_intervals is a list of lists (or list of 2-tuples)
#
# params_to_dist must take a LIST of params, even if it only needs one parameter.
def __uniform_dist_over_parametrized_credal_set_helper__(\
        param_intervals, params_to_dist, \
        distance_metric=total_variation_distance, \
        num_points_per_param=1001):

    this_interval = param_intervals[0]
    remaining_intervals = param_intervals[1:]
    epsilon = bigfloat.exp2(-128) * (this_interval[1] - this_interval[0])

    """
    if len(remaining_intervals) == 0:
        (dists, prob_multiples) = \
            singly_parametrized_dists_and_derivatives(\
                alt_dist_generator=params_to_dist, \
                alt_dist_param_bounds=this_interval, \
                num_param_options=num_points_per_param, \
                epsilon=epsilon)

        return (dists, prob_multiples)
    """

    FINAL = (len(remaining_intervals) == 0)

    param_min = this_interval[0]
    param_max = this_interval[1]

    param_inc = (param_max - param_min) / (num_points_per_param - 1)

    distributions = []
    param_derivative_multiples = []
    for i in range(0, num_points_per_param):
        param = param_min + ((param_max - param_min) * i) / \
                                (num_points_per_param - 1)

        if FINAL:
            dist = params_to_dist([param])
            distributions.append(dist)
        else:
            generator = (lambda y: (lambda x: params_to_dist([y] + x)))(param)
            (dists, measure_values) = \
                __uniform_dist_over_parametrized_credal_set_helper__(\
                    param_intervals=remaining_intervals, \
                    params_to_dist=generator, \
                    distance_metric=distance_metric, \
                    num_point_per_param=num_points_per_param)
            distributions += dists

        if param > param_min + (param_max - param_min) / 2:
            deriv_param = param - epsilon
        else:
            deriv_param = param + epsilon

        if FINAL:
            deriv_dist = params_to_dist([deriv_param])
            param_derivative_multiples.append(\
                distance_metric(dist, deriv_dist) / epsilon)
        else:
            # TODO: THIS MIGHT NOT BE THE SAME AS THE PAPER!!!!! UPDATE!!!!!
            generator = (lambda y: (lambda x: params_to_dist([y] + x)))(deriv_param)
            (_, deriv_measure_values) = \
                __uniform_dist_over_parametrized_credal_set_helper__(\
                    param_intervals=remaining_intervals, \
                    params_to_dist=generator, \
                    distance_metric=distance_metric, \
                    num_point_per_param=num_points_per_param)
            for i in range(0, len(measure_values)):
                mv = measure_values[i]
                dmv = derivative_measure_values[i]
                param_derivative_multipls.append(\
                    bigfloat.abs(mv - dmv) / epsilon)

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
