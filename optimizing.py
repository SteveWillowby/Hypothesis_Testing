import bigfloat
import matplotlib.pyplot as plt

# Assumes function is convex
def binary_min_finder(func, low, high, tol=bigfloat.BigFloat(2.0**(-30)), error_depth=1):
    low = bigfloat.BigFloat(low)
    high = bigfloat.BigFloat(high)
    mid = low + ((high - low) / 2.0)

    low_func = func(low)
    high_func = func(high)
    mid_func = func(mid)

    low_func >= mid_func or high_func >= mid_func

    val_func_pairs = [(low, low_func), (high, high_func), (mid, mid_func)]

    best_arg = mid
    best_func = mid_func
    if low_func < best_func:
        best_arg = low
        best_func = low_func
    if high_func < best_func:
        best_arg = high
        best_func = high_func

    iters = 0

    while high - low > tol:
        # print("  Remaining: %f" % ((high - low) - tol))
        left_mid = low + ((mid - low) / 2.0)
        right_mid = mid + ((high - mid) / 2.0)

        left_mid_func = func(left_mid)
        right_mid_func = func(right_mid)

        val_func_pairs.append((left_mid, left_mid_func))
        val_func_pairs.append((right_mid, right_mid_func))

        if left_mid_func < right_mid_func and left_mid_func < best_func:
            # print("A entails...")
            best_func = left_mid_func
            best_arg = left_mid
        elif right_mid_func < best_func:
            # print("B entails...")
            best_func = right_mid_func
            best_arg = right_mid
        else:
            pass
            # print("C is free...")

        if mid_func < left_mid_func and mid_func < right_mid_func:
            high = right_mid
            low = left_mid
            high_func = right_mid_func
            low_func = left_mid_func
        elif right_mid_func < mid_func and mid_func < left_mid_func:
            low = mid
            low_func = mid_func
            mid = right_mid
            mid_func = right_mid_func
        elif right_mid_func > mid_func and mid_func > left_mid_func:
            high = mid
            high_func = mid_func
            mid = left_mid
            mid_func = left_mid_func
        elif right_mid_func == mid_func and mid_func == left_mid_func:
            print("Wow!")
            high = mid
            high_func = mid_func
            low = mid
            low_func = mid_func
        elif right_mid_func == mid_func and mid_func < left_mid_func:
            low = mid
            low_func = mid_func
            mid = right_mid
            mid_func = right_mid_func
        elif left_mid_func == mid_func and mid_func < right_mid_func:
            high = mid
            high_func = mid_func
            mid = left_mid
            mid_func = left_mid_func
        elif right_mid_func == mid_func:
            assert mid_func > left_mid_func
            high = mid
            high_func = mid_func
            mid = left_mid
            mid_func = left_mid_func
        elif left_mid_func == mid_func:
            assert mid_func > right_mid_func
            low = mid
            low_func = mid_func
            mid = right_mid
            mid_func = right_mid_func
        else:
            assert mid_func > left_mid_func and mid_func > right_mid_func
            assert left_mid != mid and right_mid != mid

            """
            print("Error! Convexity assumption broken!")
            print("  low    \tleft_mid \tmid     \tright_mid \thigh")
            print("  %f \t%f \t%f \t%f \t%f" % (low, left_mid, mid, right_mid, high))
            print("  %f \t%f \t%f \t%f \t%f" % (low_func, left_mid_func, mid_func, right_mid_func, high_func))
            return (best_arg, best_func)
            """
            print("Error! Convexity assumption broken at a depth of %d" % error_depth)
            if float(left_mid_func) == float(right_mid_func):
                print("Results are close enough to quit.")
                if left_mid_func < right_mid_func:
                    return (left_mid, left_mid_func)
                else:
                    return (right_mid, right_mid_func)
            print("  low    \tleft_mid \tmid     \tright_mid \thigh")
            print("  %f \t%f \t%f \t%f \t%f" % (low, left_mid, mid, right_mid, high))
            print("  %f \t%f \t%f \t%f \t%f" % (low_func, left_mid_func, mid_func, right_mid_func, high_func))
            (best_arg_left, best_func_left) = binary_min_finder(func, low, mid, tol=tol, error_depth=(error_depth + 1))
            (best_arg_right, best_func_right) = binary_min_finder(func, mid, high, tol=tol, error_depth=(error_depth + 1))
            if best_func_left < best_func_right:
                return (best_arg_left, best_func_left)
            else:
                return (best_arg_right, best_func_right)

        iters += 1
    # print("Iters: %d" % iters)

    """
    val_func_pairs.sort()
    x_axis = [a for (a, b) in val_func_pairs]
    y_axis = [b for (a, b) in val_func_pairs]
    plt.plot(x_axis, y_axis)
    plt.title("Best of %f at %f" % (best_func, best_arg))
    plt.show()
    """

    return (best_arg, best_func)


# func must be convex
def min_finder_low_bound_only(func, low_arg, tol=bigfloat.BigFloat(2.0**(-30))):
    prev_prev_arg = low_arg
    prev_arg = low_arg
    curr_arg = low_arg

    prev_func = func(curr_arg)
    curr_func = prev_func

    while curr_func <= prev_func:
        prev_prev_arg = prev_arg
        prev_arg = curr_arg
        curr_arg *= 2.0

        prev_func = curr_func
        curr_func = func(curr_arg)

    # return (best_arg, best_func)
    return binary_min_finder(func, low=prev_prev_arg, high=curr_arg, tol=tol)

def max_finder_low_bound_only(func, low_arg, tol=bigfloat.BigFloat(2.0**(-30))):
    func_prime = (lambda x: -1.0 * func(x))
    (best_arg, best_func) = min_finder_low_bound_only(func_prime, low_arg, tol=tol)
    return (best_arg, -1.0 * best_func)

# Designed to search a function which is convex on a macro scale but not at a
#   micro scale.
#
# Repeatedly find the best and second-best arg/func pairs out of
#   `values_per_iteration` samples. Then re-search centered on the space halfway
#   between the best and second-best args. The width of the space is the gap
#   between the best and second-best args multiplied by `spread`. Do this for a
#   total of `iterations` iterations.
#
# Technically this allows the result to be outside the range of min_arg and
#   max_arg, so we allow `hard_min` and `hard_max`. All values tested will be
#   larger than `hard_min` and less than `hard_max`. If `hard_min_inclusive` is
#   set, the values can get as small as `hard_min`. Likewise for
#   `hard_max_inclusive`.
#
# `diagnose` produces a plot of all the points tested
def search_semi_convex_range(min_arg, max_arg, func, find="min", \
        hard_min=None, hard_max=None, \
        hard_min_inclusive=False, hard_max_inclusive=False, \
        iterations=20, values_per_iteration=100, spread=3.0, \
        diagnose=False):
    assert iterations >= 1
    assert values_per_iteration >= 5
    assert find == "min" or find == "max"
    assert spread > 1.0

    assert hard_min is None or hard_min <= min_arg
    assert hard_max is None or hard_max >= max_arg

    prev_best_arg = None
    prev_second_best_arg = None

    very_best_arg = None
    very_best_func_value = None

    if diagnose:
        diagnostic_points = {}
        orig_min_arg = min_arg
        orig_max_arg = max_arg

    for iteration in range(0, iterations):
        start_arg = min_arg
        arg_increment = (max_arg - min_arg) / (values_per_iteration - 1)

        best_arg = None
        best_func_value = None
        second_best_arg = None
        second_best_func_value = None
        for i in range(0, values_per_iteration):
            curr_arg = start_arg + i * arg_increment

            # Protect from invalid arguments to `func`
            if (not hard_min_inclusive) and not (hard_min is None):
                if curr_arg <= hard_min:
                    continue
            if (not hard_max_inclusive) and not (hard_max is None):
                if curr_arg >= hard_max:
                    continue

            curr_func_value = func(curr_arg)

            if diagnose:
                diagnostic_points[curr_arg] = curr_func_value

            if best_arg is None:
                best_arg = curr_arg
                best_func_value = curr_func_value
            else:
                if find == "min":
                    if curr_func_value < best_func_value:
                        second_best_func_value = best_func_value
                        second_best_arg = best_arg
                        best_func_value = curr_func_value
                        best_arg = curr_arg
                    elif second_best_arg is None or \
                            curr_func_value < second_best_func_value:
                        second_best_func_value = curr_func_value
                        second_best_arg = curr_arg
                else:
                    if curr_func_value > best_func_value:
                        second_best_func_value = best_func_value
                        second_best_arg = best_arg
                        best_func_value = curr_func_value
                        best_arg = curr_arg
                    elif second_best_arg is None or \
                            curr_func_value > second_best_func_value:
                        second_best_func_value = curr_func_value
                        second_best_arg = curr_arg

        if prev_best_arg is None:
            prev_best_arg = best_arg
            prev_second_best_arg = second_best_arg

            very_best_arg = best_arg
            very_best_func_value = best_func_value
        else:
            if find == "min":
                if very_best_func_value > best_func_value:
                    very_best_func_value = best_func_value
                    very_best_arg = best_arg
            else:
                if very_best_func_value < best_func_value:
                    very_best_func_value = best_func_value
                    very_best_arg = best_arg

            if prev_best_arg == best_arg and prev_second_best_arg == second_best_arg:
                print("Ending a search %d iterations early because nothing is changing." % \
                    (iterations - (iteration + 1)))
                break

            prev_best_arg = best_arg
            prev_second_best_arg = second_best_arg

        if best_arg > second_best_arg:
            higher_arg = best_arg
            lower_arg = second_best_arg
        else:
            higher_arg = second_best_arg
            lower_arg = best_arg

        new_center = lower_arg + (higher_arg - lower_arg) / 2.0
        half_new_width = ((higher_arg - lower_arg) / 2.0) * spread

        new_min = new_center - half_new_width
        if not hard_min is None and new_min < hard_min:
            if hard_min_inclusive:
                new_min = hard_min
            else:
                # Move VERY close to new_min. Can get closer each time.
                new_min = (1000.0 * hard_min + lower_arg) / 1001.0
        new_max = new_center + half_new_width
        if not hard_max is None and new_max > hard_max:
            if hard_max_inclusive:
                new_max = hard_max
            else:
                new_max = (1000.0 * hard_max + higher_arg) / 1001.0

        min_arg = new_min
        max_arg = new_max
        if diagnose and iteration + 1 < iterations:
            print("Iteration %d will use range: (%s, %s)" % (iteration + 2, min_arg, max_arg))

    if diagnose:
        diagnostic_points = \
            [(arg, value) for (arg, value) in diagnostic_points.items()]
        diagnostic_points.sort()
        x = [a for (a, b) in diagnostic_points]
        y = [b for (a, b) in diagnostic_points]
        plt.plot(x, y)
        plt.suptitle("%simizing `func` on the range (%f, %f)" % \
            (find, orig_min_arg, orig_max_arg))
        plt.title("Iterations: %d, Points per Iteration: %d, Spread: %f" % \
            (iterations, values_per_iteration, spread))
        plt.xlabel("Unknown X Axis")
        plt.ylabel("`func` Values")
        plt.show()

    return (very_best_arg, very_best_func_value)
